"""
需要草图图片的 x, y 范围均在 [-1, 1] 之间
"""
from tqdm.auto import tqdm
from torch.nn import Module
import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    """
    将 t 中数值作为索引，提取 a 中对应数值，并将其维度扩充至与 x_shape 维度数相同
    :param a: 一维数组 [n, ]
    :param t: 一维数组，表示批量内的全部时间步 [bs, ]
    :param x_shape: 目标数组的大小 [bs, channel, n_skh_pnt]
    :return:
    """
    bs = t.shape[0]
    out = a.gather(-1, t)  # 从张量 a 的 -1 维度提取 t 索引数组对应值，out 与 t 形状相同
    blank_channel = (1,) * (len(x_shape) - 1)  # -> [2, ], 元组乘法表示将其重复一定次数
    out = out.reshape(bs, *blank_channel)  # -> [bs, 1, 1], len = len(x_shape)
    return out


class GaussianDiffusion(Module):
    def __init__(self, model, pnt_channel, n_skh_pnt, timesteps=2000):  # 1000
        super().__init__()

        self.model = model
        self.pnt_channel = pnt_channel
        self.n_skh_pnt = n_skh_pnt
        self.timesteps = timesteps

        # 基本参数数组
        betas = torch.linspace(0.001, 0.02, timesteps, dtype=torch.float64)  # 0.0001, 0.2
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
        :param batch_size: 生成图片数
        :return:
        """
        # 获取纯高斯噪音
        img = torch.randn((batch_size, self.pnt_channel, self.n_skh_pnt), device=self.device)

        # 倒着遍历所有时间步，从噪音还原图片
        for t in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.inference_x_t_minus1(img, t)

        return img

    def noise_pred_loss(self, x_0, t):
        """
        输入原始图片和时间步t，计算噪音预测的loss
        :param x_0: 原始图片 [bs, channel, n_skh_pnt]
        :param t: 当前时间步 [bs, ]
        :return: 模型在当前时间步预测噪音的损失
        """
        # 生成正态分布噪音
        noise = torch.randn_like(x_0)

        # 获取 t 时间步下加噪后的图片
        x_t = extract(self.sqrt_1minus_alphas_bar, t, x_0.shape) * noise + extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0

        # 获取模型预测的原始图片
        x_0_pred = self.model(x_t, t)

        return F.mse_loss(x_0_pred, noise)

        # 计算噪音
        # noise_pred = extract(self.sqrt_recip_1minus_alphas_bar, t, x_t.shape) * x_t - extract(self.sqrt_recip_recip_alphas_bar_minus1, t, x_t.shape) * x_0_pred
        # return F.mse_loss(noise_pred, noise) + F.mse_loss(x_0_pred, x_0)

    def forward(self, img):
        """
        :param img: 原始草图 [bs, pnt_channel, n_skh_pnt]
        :return: 噪音预测损失
        """
        bs = img.size(0)
        device = img.device

        # 为每个批量的图片生成时间步
        t = torch.randint(0, self.timesteps, (bs,), device=device).long()  # -> [bs, ]

        return self.noise_pred_loss(img, t)


