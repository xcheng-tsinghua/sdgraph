from tqdm.auto import tqdm
from torch.nn import Module
import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img):
    """
    将转换到[0, 1]的图片还原，用于计算loss
    :param img:
    :return:
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    将图片转换到[0, 1]用于可视化
    :param t:
    :return:
    """
    return (t + 1) * 0.5


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        image_size=32,  # 生成图片大小
        timesteps=1000,  # diffusion 时间步
    ):
        super().__init__()

        self.model = model
        self.channels = self.model.channels
        self.image_size = self.model.n_pnts
        self.num_timesteps = timesteps

        # 基本参数数组
        betas = torch.linspace(0.0001, 0.2, timesteps, dtype=torch.float64)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 从原始图片生成第t时间步加噪图片的系数
        register_buffer('sqrt_alphas_bar', alphas_bar.sqrt())
        register_buffer('sqrt_1minus_alphas_bar', (1. - alphas_bar).sqrt())

        # 根据当前时间步的图片和（预测的）噪音，推理原始图片的公式中的系数。reciprocal: 倒数
        register_buffer('sqrt_recip_alphas_bar', (1. / alphas_bar).sqrt())
        register_buffer('sqrt_recip_alphas_bar_minus1', (1. / alphas_bar - 1).sqrt())

        # 根据 x_t 和 x_0 计算噪音的系数
        register_buffer('sqrt_recip_1minus_alphas_bar', (1. / (1. - alphas_bar)).sqrt())
        register_buffer('sqrt_recip_recip_alphas_bar_minus1', (1. / ((1. / alphas_bar) - 1)).sqrt())

        # 从当前步推理前一步图片所属的正态分布的系数
        register_buffer('back_inf_std', ((betas * (1. - alphas_bar_prev) / (1. - alphas_bar)).clamp(min=1e-20)).sqrt())
        register_buffer('back_inf_mean_coef1', (alphas.sqrt() * (1. - alphas_bar_prev) / (1. - alphas_bar)))
        register_buffer('back_inf_mean_coef2', (alphas_bar_prev.sqrt() * betas / (1. - alphas_bar)))

    @property
    def device(self):
        return self.sqrt_alphas_bar.device

    @torch.inference_mode()
    def inference_mean_std(self, x_t, t):
        """
        通过 x_t 和 t 推导上一步中服从分布的均值和方差
        :param x_t:
        :param t:
        :return:
        """
        noise = self.model(x_t, t)

        x_0 = extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - extract(self.sqrt_recip_alphas_bar_minus1, t, x_t.shape) * noise
        x_0.clamp_(-1., 1.)

        model_mean = extract(self.back_inf_mean_coef2, t, x_t.shape) * x_0 + extract(self.back_inf_mean_coef1, t, x_t.shape) * x_t
        model_std = extract(self.back_inf_std, t, x_t.shape)

        return model_mean, model_std

    @torch.inference_mode()
    def inference_x_t_minus1(self, x_t, t: int):
        """
        使用当前时间步的图片生成上一步图片
        :param x_t: 当前时间步的图片
        :param t: 当前时间步
        :return:
        """
        bs = x_t.size(0)

        # 生成全为 t 的张量，大小：[bs, ]
        batched_times = torch.full((bs,), t, device=self.device, dtype=torch.long)

        model_mean, model_std = self.inference_mean_std(x_t, batched_times)

        if t > 0:
            pred_img = torch.normal(mean=model_mean, std=model_std)
        else:
            pred_img = model_mean

        return pred_img

    @torch.inference_mode()
    def sample(self, batch_size=20):
        """
        从扩散模型获取图片
        :param batch_size:
        :param return_all_timesteps:
        :return:
        """
        # 获取纯高斯噪音
        img = torch.randn((batch_size, self.channels, self.image_size), device=self.device)

        # 倒着遍历所有时间步，从噪音还原图片
        for t in tqdm(reversed(range(self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.inference_x_t_minus1(img, t)

        # 将图片从[-1, 1]转化到[0, 1]，通过 (img + 1) / 2
        # img = unnormalize_to_zero_to_one(img)

        return img

    def noise_pred_loss(self, x_0, t):
        """
        输入原始图片和时间步t，计算噪音预测的loss
        :param x_0: 原始图片
        :param t: 当前时间步
        :return: 模型在当前时间步预测噪音的损失
        """
        # -> x_start: [bs, channel, height, width]
        # -> t: [bs, ]

        # 生成正态分布噪音
        noise = torch.randn_like(x_0)

        # 获取 t 时间步下加噪后的图片
        x_t = extract(self.sqrt_1minus_alphas_bar, t, x_0.shape) * noise + extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0

        # 获取模型预测的原始图片
        x_0_pred = self.model(x_t, t)
        # x_0_pred.clamp_(-1., 1.)  # ----------------------

        return F.mse_loss(x_0_pred, x_0)

        # 计算噪音
        # noise_pred = extract(self.sqrt_recip_1minus_alphas_bar, t, x_t.shape) * x_t - extract(self.sqrt_recip_recip_alphas_bar_minus1, t, x_t.shape) * x_0_pred
        #
        # return F.mse_loss(noise_pred, noise) + F.mse_loss(x_0_pred, x_0)

    def forward(self, img):
        # 输入img为原始图片

        bs = img.size(0)
        device = img.device

        # -> [bs, ], 为每个批量的图片生成时间步
        t = torch.randint(0, self.num_timesteps, (bs,), device=device).long()

        # 将 [0, 1] 的图片转化为 [-1, 1]，通过 img*2 - 1
        # img = normalize_to_neg_one_to_one(img)

        return self.noise_pred_loss(img, t)





