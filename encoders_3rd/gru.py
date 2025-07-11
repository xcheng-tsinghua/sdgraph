import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.utils import MLP


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # 如果是双向GRU，输出维度翻倍
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: [bs, n_pnts, 3]
        _, h_n = self.gru(x)  # h_n: [num_layers * num_directions, bs, hidden_dim]

        if self.bidirectional:
            # 拼接最后一层正向和反向的 hidden state
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [bs, hidden_dim * 2]
        else:
            h_last = h_n[-1]  # [bs, hidden_dim]

        return h_last  # [bs, out_dim]


class GRU(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        print('create GRU classifier')
        self.gru_encoder = GRUEncoder()

        emb_l0 = 512
        emb_inc = (n_classes / emb_l0) ** (1 / 3)
        emb_l1 = int(emb_l0 * emb_inc)
        emb_l2 = int(emb_l0 * emb_inc * emb_inc)
        emb_l3 = n_classes

        self.cls_head = MLP(0, (emb_l0, emb_l1, emb_l2, emb_l3))

    def forward(self, x, mask=None):
        """
        :param x: [bs, len_seq, 3]
        :param mask: 占位用
        :return:
        """
        emb = self.gru_encoder(x)
        emb = self.cls_head(emb)
        emb = F.log_softmax(emb, dim=1)

        return emb


if __name__ == '__main__':
    atensor = torch.rand(9, 100, 3).cuda()
    anet = GRU(7).cuda()

    aout = anet(atensor)

    print(aout.size())








