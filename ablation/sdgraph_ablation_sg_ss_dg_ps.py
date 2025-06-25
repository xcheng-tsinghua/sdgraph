"""
用于消融实验
SG + SS + DG + PS
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

import global_defs
from encoders import utils as eu
from encoders.sdgraph_ablation_sg_ss import Ablation_SG_SS_Embedding
from encoders.sdgraph_ablation_dg_ps import Ablation_DG_PS_Embedding


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, channel_in=2, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('sdgraph Ablation (SG + SS) + (DG + PS)')

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.channel_in = channel_in

        # 各层特征维度
        sparse_l0 = 32 + 16
        sparse_l1 = 128 + 64
        sparse_l2 = 512 + 256

        dense_l0 = 32
        dense_l1 = 128
        dense_l2 = 512

        self.embedding_sg = Ablation_SG_SS_Embedding(sparse_l0, sparse_l1, sparse_l2, channel_in, dropout, n_stk, n_stk_pnt)
        self.embedding_dg = Ablation_DG_PS_Embedding(dense_l0, dense_l1, dense_l2, channel_in, dropout, n_stk, n_stk_pnt)

        # 利用输出特征进行分类
        sparse_glo = sparse_l0 + sparse_l1 + sparse_l2
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / (sparse_glo + dense_glo)) ** (1 / 3)

        out_l0 = sparse_glo + dense_glo
        out_l1 = int(out_l0 * out_inc)
        out_l2 = int(out_l1 * out_inc)
        out_l3 = n_class

        self.linear = eu.MLP(dimension=0,
                             channels=(out_l0, out_l1, out_l2, out_l3),
                             final_proc=False,
                             dropout=dropout)

    def forward(self, xy, mask=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param mask: 占位用
        :return: [bs, n_classes]
        """

        sg_emb = self.embedding_sg(xy)
        dg_emb = self.embedding_dg(xy)

        all_fea = torch.cat([sg_emb, dg_emb], dim=1)

        # 利用全局特征分类
        cls = self.linear(all_fea)
        cls = F.log_softmax(cls, dim=1)

        return cls


def test():
#     bs = 3
#     atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
#     t1 = torch.randint(0, 1000, (bs,)).long().cuda()
#
#     # classifier = SDGraphSeg2(2, 2).cuda()
#     # cls11 = classifier(atensor, t1)
#
#     classifier = SDGraphCls2(10).cuda()
#     cls11 = classifier(atensor)
#
#     print(cls11.size())
#
#     print('---------------')


    bs = 3
    atensor = torch.rand([bs, global_defs.n_stk, global_defs.n_stk_pnt, 3])
    t1 = torch.randint(0, 1000, (bs,)).long()

    classifier_cls = SDGraphCls(10, 3)
    cls12 = classifier_cls(atensor)
    print(cls12.size())

    n_paras = eu.count_parameters(classifier_cls)
    print(f'model parameter count: {n_paras}')

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')

    # dense_fea = einops.rearrange(dense_fea, 'b c s sp -> b s (sp c)')
    #
    # dense_fea = dense_fea.permute(0, 2, 3, 1)
    # dense_fea = dense_fea.reshape(dense_fea.size(0), dense_fea.size(1), -1)





