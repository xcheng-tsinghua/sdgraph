import torch
import torch.nn as nn
import torch.nn.functional as F

import encoders.utils as utils


class AttentionEncoder(nn.Module):
    def __init__(self, emb_in, emb_out, n_token, n_head=4):
        """
        multihead attention
        :param n_channel: input channel
        :param n_head: n_head, assert n_channel % n_head == 0
        """
        super().__init__()
        self.n_token = n_token

        self.multihead_attention = nn.MultiheadAttention(embed_dim=emb_in, num_heads=n_head, dropout=0.4)
        self.linear = utils.full_connected_conv1d(channels=[emb_in, int((emb_in * emb_out) ** 0.5), emb_out], final_proc=True)

    def forward(self, fea):
        """
        :param fea: [bs, n_channel, n_token]
        :return: updated fea: [bs, n_channel, n_token]
        """
        n_stroke = fea.size()[2]
        assert n_stroke == self.n_token

        # -> [n_token, bs, n_channel]
        fea = fea.permute(2, 0, 1)

        # output -> [n_token, bs, n_channel]
        output, attention_weights = self.multihead_attention(fea, fea, fea)

        # -> [bs, n_channel, n_token]
        output = output.permute(1, 2, 0)

        # -> [bs, n_channel, n_token]
        output = self.linear(output)

        return output


class Attention(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.f1 = utils.full_connected_conv1d(channels=[2, 4, 8, 16, 32, 64, 128, 256])
        self.encoder = AttentionEncoder(128, 256, 30*34)
        self.f2 = utils.full_connected([256, 128, 64, 32, n_classes])

    def forward(self, xy):
        # -> [bs, 2, n_points]
        fea = self.f1(xy)

        # -> [bs, emb, n_points]
        # fea = self.encoder(fea)

        # -> [bs, emb]
        fea = torch.max(fea, dim=2)[0]

        fea = self.f2(fea)
        fea = F.log_softmax(fea, dim=1)

        return fea


if __name__ == '__main__':
    bs = 32
    channel = 64
    n_token = 128
    input_tensor = torch.randn(bs, channel, n_token)

    attenlatyer = AttentionEncoder(channel, 4)

    outs = attenlatyer(input_tensor)
    print("Output shape:", outs.shape)

