import einops
import timm
import torch.nn as nn
import torch
from collections import OrderedDict

from encoders.utils import MLP


class ImageEncoder_ULIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.image_projection = nn.Parameter(torch.empty(768, 512))

    # @torch.inference_mode()
    def forward(self, image):
        x = self.vision_model(image)
        x = x @ self.image_projection

        return x


class VITFinetune(nn.Module):
    """
    注意预训练的权重和预测头均会训练
    """
    def __init__(self, channel_out=512, root_ckpt='./model_trained/weight_image_encoder.pth'):
        super().__init__()
        print('create vit finetune')

        self.image_encoder_pretrained = ImageEncoder_ULIP()
        try:
            self.image_encoder_pretrained.load_state_dict(torch.load(root_ckpt), strict=True)
        except:
            raise ValueError('can not load pretrained model weight: ', root_ckpt)

        self.mlp = MLP(0, (512, int((512 * channel_out) ** 0.5), channel_out), final_proc=False)

    # @torch.inference_mode()
    def forward(self, image):
        """

        :param image: [bs, c, w, h]
        :return:
        """
        if len(image.size()) == 3:
            image = einops.repeat(image, 'b w h -> b c w h', c=3)

        self.image_encoder_pretrained = self.image_encoder_pretrained.eval()
        with torch.no_grad():
            fea = self.image_encoder_pretrained(image)

        fea = self.mlp(fea)
        return fea


def save_weights_from_all():
    ckpt = torch.load('weights_all.pt', map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    image_encoder = ImageEncoder_ULIP().cuda()

    image_encoder.load_state_dict(state_dict, strict=False)

    torch.save(image_encoder.state_dict(), 'weight_image_encoder.pth')


def create_pretrained_VIT(root_ckpt: str = './model_trained/weight_image_encoder.pth'):
    print('create pretrained image encoder, load weight from ' + root_ckpt)

    image_encoder_pretrained = ImageEncoder_ULIP()

    try:
        image_encoder_pretrained.load_state_dict(torch.load(root_ckpt), strict=True)
    except:
        raise ValueError('can not load pretrained model weight: ', root_ckpt)

    # 设为评估模式
    image_encoder_pretrained = image_encoder_pretrained.eval()

    # 禁用梯度计算，提升速度
    image_encoder_pretrained.requires_grad_(False)

    return image_encoder_pretrained


def test():
    amodel = ImageEncoder_ULIP()
    emb = torch.rand(9, 3, 224, 224)

    out = amodel(emb)
    print(out.size())


if __name__ == '__main__':
    # test()

    # save_weights_from_all()

    atensor = torch.rand(5, 3, 512, 512)

    amodel = VITFinetune(10)
    print(amodel(atensor).size())


    pass


