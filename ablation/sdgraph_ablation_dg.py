"""
用于消融实验
DG
仅Dense Graph
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

import global_defs
from encoders import utils as eu


class GCNEncoder(nn.Module):
    """
    实际上是 DGCNN Encoder
    """
    def __init__(self, emb_in, emb_out, n_near=10):
        super().__init__()
        self.n_near = n_near

        emb_inc = (emb_out / (4 * emb_in)) ** 0.25
        emb_l1_0 = emb_in * 2
        emb_l1_1 = int(emb_l1_0 * emb_inc)
        emb_l1_2 = int(emb_l1_0 * emb_inc ** 2)

        emb_l2_0 = emb_l1_2 * 2
        emb_l2_1 = int(emb_l2_0 * emb_inc)
        emb_l2_2 = emb_out

        emb_l3_0 = emb_l2_2 + emb_l1_2
        emb_l3_1 = int(((emb_out / emb_l3_0) ** 0.5) * emb_l3_0)
        emb_l3_2 = emb_out

        self.conv1 = eu.MLP(dimension=2,
                            channels=(emb_l1_0, emb_l1_1, emb_l1_2),
                            final_proc=True,
                            dropout=0.0)

        self.conv2 = eu.MLP(dimension=2,
                            channels=(emb_l2_0, emb_l2_1, emb_l2_2),
                            final_proc=True,
                            dropout=0.0)

        self.conv3 = eu.MLP(dimension=1,
                            channels=(emb_l3_0, emb_l3_1, emb_l3_2),
                            final_proc=True,
                            dropout=0.0)

    def forward(self, x):
        # x: [bs, channel, n_token]

        # -> [bs, emb, n_token, n_neighbor]
        x = self.get_graph_feature(x, k=self.n_near)
        x = self.conv1(x)

        # -> [bs, emb, n_token]
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x1, k=self.n_near)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # -> [bs, emb, n_token]
        x = torch.cat((x1, x2), dim=1)

        # -> [bs, emb, n_token]
        x = self.conv3(x)

        return x

    @staticmethod
    def get_graph_feature(x, k):
        """
        找到x中每个点附近的k个点，然后计算中心点到附近点的向量，然后拼接在一起
        :param x: [bs, channel, n_point]
        :param k:
        :return: [bs, channel, n_point, n_near]
        """
        x = x.permute(0, 2, 1)  # -> [bs, n_point, channel]

        # step1: 通过knn计算附近点坐标，邻近点不包含自身
        knn_idx = eu.knn(x, k)  # (batch_size, num_points, k)

        # step2: 找到附近点
        neighbors = eu.index_points(x, knn_idx)  # -> [bs, n_point, k, channel]

        # step3: 计算向量
        cen_to_nei = neighbors - x.unsqueeze(2)  # -> [bs, n_point, k, channel]

        # step4: 拼接特征
        feature = torch.cat([cen_to_nei, x.unsqueeze(2).repeat(1, 1, k, 1)], dim=3)  # -> [bs, n_point, k, channel]
        feature = feature.permute(0, 3, 1, 2)  # -> [bs, channel, n_point, k]

        return feature


class PointToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph'
    使用一维卷积及最大池化方式
    """
    def __init__(self, point_dim, sparse_out, dropout=0.0):
        super().__init__()

        # 将 DGraph 的数据转移到 SGraph
        mid_dim = int((point_dim * sparse_out) ** 0.5)
        self.point_increase = nn.Sequential(
            nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
            nn.BatchNorm2d(mid_dim),
            eu.activate_func(),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
            nn.BatchNorm2d(sparse_out),
            eu.activate_func(),
            nn.Dropout2d(dropout),
        )

    def forward(self, xy):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = xy.permute(0, 3, 1, 2)

        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = self.point_increase(xy)

        # -> [bs, emb, n_stk]
        xy = torch.max(xy, dim=3)[0]
        # assert xy.size(2) == self.n_stk

        return xy


class PointToDense(nn.Module):
    """
    利用点坐标生成 dense graph
    使用DGCNN直接为每个点生成对应特征
    """
    def __init__(self, point_dim, emb_dim, n_near=10):
        super().__init__()
        self.encoder = GCNEncoder(point_dim, emb_dim, n_near)

    def forward(self, xy,):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        bs, n_stk, n_stk_pnt, channel = xy.size()

        xy = einops.rearrange(xy, 'b s sp c -> b c (s sp)')
        dense_emb = self.encoder(xy)

        dense_emb = dense_emb.view(bs, dense_emb.size(1), n_stk, n_stk_pnt)
        return dense_emb


class SparseUpdate(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self, sp_in, sp_out, n_near=2):
        super().__init__()
        self.encoder = GCNEncoder(sp_in, sp_out, n_near)

    def forward(self, sparse_fea):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        sparse_fea = self.encoder(sparse_fea)
        return sparse_fea


class DenseUpdate(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self, dn_in, dn_out, n_near=10, dropout=0.0):
        super().__init__()
        # 先利用时序进行更新
        # self.temporal_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=dn_in, out_channels=dn_in, kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(dn_in),
        #     eu.activate_func(),
        #     nn.Dropout2d(dropout),
        # )

        # 再利用GCN进行更新
        self.encoder = GCNEncoder(dn_in, dn_out, n_near)

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """

        bs, emb, n_stk, n_stk_pnt = dense_fea.size()

        # dense_fea = self.temporal_encoder(dense_fea)

        dense_fea = dense_fea.view(bs, emb, n_stk * n_stk_pnt)

        dense_fea = self.encoder(dense_fea)

        dense_fea = dense_fea.view(bs, dense_fea.size(1), n_stk, n_stk_pnt)

        return dense_fea


class DownSample(nn.Module):
    """
    对sdgraph同时进行下采样
    该模块处理后笔划数和笔划中的点数同时降低为原来的1/2
    """
    def __init__(self, dn_in, dn_out, dropout=0.4):
        """
        :param dn_in:
        :param dn_out:
        :param dropout:
        """
        super().__init__()

        # dense graph 下采样及特征更新
        self.dn_downsamp = nn.Sequential(
            nn.Conv2d(
                in_channels=dn_in,  # 输入通道数 (RGB)
                out_channels=dn_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dn_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        """
        # ------- 对dense fea进行下采样 -------
        # 进行下采样
        dense_fea = self.dn_downsamp(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt // 2]
        return dense_fea


class SparseToDense(nn.Module):
    """
    将sgraph转移到dgraph
    直接拼接到该笔划对应的点
    """
    def __init__(self, sparse_in, dense_in, dropout):
        super().__init__()

        self.encoder = eu.MLP(
            dimension=2,
            channels=(sparse_in + dense_in, dense_in),
            dropout=dropout,
            final_proc=True
        )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk * n_stk_pnt]
        """
        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, dense_fea.size(3))
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = self.encoder(union_dense)

        return union_dense


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self, sparse_in, dense_in, dropout):
        super().__init__()

        # 先利用时序进行更新
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3), padding=(0, 1)),  # 输入 = 输出
            nn.BatchNorm2d(dense_in),
            eu.activate_func(),
            nn.Dropout2d(dropout),
        )

        self.encoder = eu.MLP(
            dimension=1,
            channels=(sparse_in + dense_in, sparse_in),
            dropout=dropout,
            final_proc=True
        )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt]
        dense_fea = self.temporal_encoder(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = dense_fea.max(3)[0]

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)
        union_sparse = self.encoder(union_sparse)

        return union_sparse


class SDGraphEncoder(nn.Module):
    """
    包含笔划及笔划中的点层级的下采样与上采样
    """
    def __init__(self,
                 dense_in, dense_out,
                 dn_near=10,
                 dropout=0.2
                 ):
        """
        :param dense_in:
        :param dense_out:
        :param dn_near: 更新 dgraph 时个GCN中邻近点数目
        :param dropout:
        """
        super().__init__()

        self.dense_update = DenseUpdate(dense_in, dense_out, dn_near, dropout)  # 此处dropout效果未测试

    def forward(self, dense_fea, stk_coor=None):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: 笔划坐标 [bs, n_stk, emb]
        :return:
        """

        bs, _, n_stk, n_stk_pnt = dense_fea.size()

        # 信息更新
        union_dense = self.dense_update(dense_fea)

        return union_dense


class Ablation_DG_Embedding(nn.Module):
    def __init__(self, dense_l0, dense_l1, dense_l2, channel_in, dropout):
        """
        :param n_class: 总类别数
        """
        super().__init__()

        self.channel_in = channel_in

        # 生成笔划坐标
        self.point_to_dense = PointToDense(channel_in, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(dense_l0, dense_l1,
                                  )

        self.sd2 = SDGraphEncoder(dense_l1, dense_l2,
                                  )

    def forward(self, xy, mask=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param mask: 占位用
        :return: [bs, n_classes]
        """
        bs, n_stk, n_stk_pnt, channel = xy.size()

        # 生成初始 dense graph
        dense_graph0 = self.point_to_dense(xy)

        # 交叉更新数据
        dense_graph1 = self.sd1(dense_graph0)
        dense_graph2 = self.sd2(dense_graph1)

        # 提取全局特征
        dense_glo0 = dense_graph0.amax((2, 3))
        dense_glo1 = dense_graph1.amax((2, 3))
        dense_glo2 = dense_graph2.amax((2, 3))

        all_fea = torch.cat([dense_glo0, dense_glo1, dense_glo2], dim=1)

        return all_fea


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, channel_in=2, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('sdgraph ablation DG')

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.channel_in = channel_in

        # 各层特征维度
        dense_l0 = 32
        dense_l1 = 128
        dense_l2 = 512

        self.embedding = Ablation_DG_Embedding(dense_l0, dense_l1, dense_l2, channel_in, dropout)

        # 利用输出特征进行分类
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / dense_glo) ** (1 / 3)

        out_l0 = dense_glo
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
        all_fea = self.embedding(xy)

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





