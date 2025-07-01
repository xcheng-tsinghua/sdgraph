"""
用于验证提出的目标学习内容是否有效
ISI：inter stroke information
通过断开 SGraph 中的节点连接
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
        self.encoder = eu.MLP(1, (sp_in, int((sp_in * sp_out) ** 0.5), sp_out))

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
    def __init__(self, sp_in, sp_out, dn_in, dn_out, n_stk, n_stk_pnt, sp_near=2, dropout=0.4):
        """
        :param sp_in:
        :param sp_out:
        :param dn_in:
        :param dn_out:
        :param n_stk:
        :param n_stk_pnt:
        :param sp_near: 在对 sparse graph 下采样过程中，获取中心点特征时邻近点的数量
        :param dropout:
        """
        super().__init__()
        self.sp_near = sp_near

        # 笔划数及单个笔划内的点数
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # sparse graph 特征更新层
        self.sp_conv = eu.MLP(dimension=2,
                              channels=(sp_in * 2, sp_out),
                              dropout=dropout,
                              final_proc=True)

        # dense graph 特征更新层
        self.dn_conv = eu.MLP(dimension=3,
                              channels=(dn_in * 2, dn_out),
                              dropout=dropout,
                              final_proc=True)

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

    def forward(self, sparse_fea, dense_fea, stk_coor, n_stk_center=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: [bs, n_stk, emb]
        :param n_stk_center: 笔划下采样过程中的笔划采样点数
        """
        bs, emb, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt

        n_stk = sparse_fea.size(2)
        assert n_stk == self.n_stk == stk_coor.size(1)

        # ------- 对sgraph进行下采样 -------
        # 使用FPS采样，获得获得中心点索引
        fps_idx = eu.fps(stk_coor, n_stk_center)  # -> [bs, n_stk_center]

        # 根据中心点索引，获取中心点特征
        sparse_center_fea = eu.index_points(sparse_fea, fps_idx, True)
        assert sparse_center_fea.size(2) == n_stk_center

        # step1: --- 获得中心点附近的点的特征 ---
        # 获取全局knn索引
        knn_idx_all = eu.knn(stk_coor, self.sp_near)  # -> [bs, n_stk, self.sp_near]
        assert knn_idx_all.size(2) == self.sp_near

        # 获取中心点的附近的点索引
        knn_idx_fps = eu.index_points(knn_idx_all, fps_idx)  # -> [bs, n_stk // 2, self.sp_near]
        assert knn_idx_fps.size(1) == n_stk_center

        # 根据附近点索引获取附近点特征
        sparse_neighbor_fea = eu.index_points(sparse_fea, knn_idx_fps, True)  # -> [bs, emb, n_stk // 2, sp_near]

        # step2: --- 获取局部区域特征 ---
        # 获取辅助特征，即从中心点到邻近点的向量
        sparse_assist_fea = sparse_neighbor_fea - sparse_center_fea.unsqueeze(3)  # -> [bs, emb, n_stk // 2, sp_near]

        # 拼接中心特征和辅助特征
        sparse_center_fea = einops.repeat(sparse_center_fea, 'b c s -> b c s n', n=self.sp_near)
        sparse_fea = torch.cat([sparse_assist_fea, sparse_center_fea], dim=1)  # -> [bs, emb, n_stk // 2, sp_near]

        # step3: --- 更新特征 ---
        sparse_fea = self.sp_conv(sparse_fea)  # -> [bs, emb, n_stk // 2, sp_near]
        sparse_fea = sparse_fea.max(3)[0]  # -> [bs, emb, n_stk // 2]
        assert sparse_fea.size(2) == n_stk_center

        # step4: --- 获取采样后的笔划坐标 ---
        stk_coor_sampled = eu.index_points(stk_coor, fps_idx)  # -> [bs, n_stk // 2, emb]
        assert stk_coor_sampled.size(1) == n_stk_center

        # ------- 对dense fea进行下采样 -------
        # 先找到 fps 获取的对应的笔划
        dense_fea = einops.rearrange(dense_fea, 'b c s sp -> b s (sp c)')

        dense_center_fea = eu.index_points(dense_fea, fps_idx)  # -> [bs, n_stk // 2, n_stk_pnt * emb]

        # 找到对应笔划附近的笔划
        dense_neighbor_fea = eu.index_points(dense_fea, knn_idx_fps)  # -> [bs, n_stk // 2, sp_near, n_stk_pnt * emb]

        # 获取辅助特征，即从中心点到邻近点的向量
        dense_assist_fea = dense_neighbor_fea - dense_center_fea.unsqueeze(2)  # -> [bs, n_stk // 2, sp_near, n_stk_pnt * emb]

        # 拼接中心特征和辅助特征
        dense_center_fea = einops.repeat(dense_center_fea, 'b s c -> b s n c', n=self.sp_near)  # -> [bs, n_stk // 2, sp_near, n_stk_pnt * emb]
        dense_fea = torch.cat([dense_assist_fea, dense_center_fea], dim=3)  # -> [bs, n_stk // 2, sp_near, n_stk_pnt * emb]
        dense_fea = einops.rearrange(dense_fea, 'b s n (sp c) -> b c s sp n', sp=self.n_stk_pnt)  # -> [bs, emb, n_stk // 2, n_stk_pnt, sp_near]

        # 更新特征
        dense_fea = self.dn_conv(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt, sp_near]
        dense_fea = dense_fea.max(4)[0]  # -> [bs, emb, n_stk // 2, n_stk_pnt]
        assert dense_fea.size(3) == self.n_stk_pnt

        # 进行下采样
        dense_fea = self.dn_downsamp(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt // 2]
        return sparse_fea, dense_fea, stk_coor_sampled


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
                 sparse_in, sparse_out, dense_in, dense_out,
                 n_stk, n_stk_pnt,
                 n_stk_center, n_stk_pnt_center,
                 sp_near=2, dn_near=10,
                 dropout=0.2
                 ):
        """
        :param sparse_in: 输入维度
        :param sparse_out: 输出维度
        :param dense_in:
        :param dense_out:
        :param n_stk: 笔划数
        :param n_stk_pnt: 每个笔划中的点数
        :param n_stk_center: 笔划下采样过程中的笔划采样点数
        :param n_stk_pnt_center: 仅用作占位, 笔划上的点下采样至原来的一半，上采样至原来的两倍
        :param sp_near: 更新 sgraph 时个GCN中邻近点数目
        :param dn_near: 更新 dgraph 时个GCN中邻近点数目
        :param dropout:
        """
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.n_stk_center = n_stk_center

        self.dense_to_sparse = DenseToSparse(sparse_in, dense_in, dropout)  # 此处dropout之前测试不能设为零
        self.sparse_to_dense = SparseToDense(sparse_in, dense_in, dropout)  # 此处dropout之前测试不能设为零

        self.sparse_update = SparseUpdate(sparse_in, sparse_out, sp_near)
        self.dense_update = DenseUpdate(dense_in, dense_out, dn_near, dropout)  # 此处dropout效果未测试

        self.sample = DownSample(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout=dropout)

    def forward(self, sparse_fea, dense_fea, stk_coor=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: 笔划坐标 [bs, n_stk, emb]
        :return:
        """

        bs, emb, n_stk = sparse_fea.size()
        assert n_stk == self.n_stk

        _, _, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt

        # 信息交换
        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)
        union_dense = self.sparse_to_dense(sparse_fea, dense_fea)

        # 信息更新
        union_sparse = self.sparse_update(union_sparse)
        union_dense = self.dense_update(union_dense)

        # 采样
        union_sparse, union_dense, stk_coor_sampled = self.sample(union_sparse, union_dense, stk_coor, self.n_stk_center)
        if stk_coor_sampled is not None:
            assert stk_coor_sampled.size(0) == bs and (stk_coor_sampled.size(1) == self.n_stk_center or stk_coor_sampled.size(1) == self.n_stk * 2)

        return union_sparse, union_dense, stk_coor_sampled


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, channel_in=2, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('sdgraph evaluate the efficient of Inter Stroke Information, removed the connection when SGraph update')

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

        # 生成笔划坐标
        self.point_to_stk_coor = PointToSparse(channel_in, sparse_l0)

        # 生成初始 sdgraph
        self.point_to_sparse = PointToSparse(channel_in, sparse_l0)
        self.point_to_dense = PointToDense(channel_in, dense_l0)

        # 利用 sdgraph 更新特征
        d_down_stk = (self.n_stk - 3) // 2
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  self.n_stk, self.n_stk_pnt,
                                  self.n_stk - d_down_stk, self.n_stk_pnt // 2,
                                  dropout=dropout
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  self.n_stk - d_down_stk, self.n_stk_pnt // 2,
                                  self.n_stk - d_down_stk * 2, self.n_stk_pnt // 4,
                                  dropout=dropout
                                  )

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
        bs, n_stk, n_stk_pnt, channel = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel == self.channel_in

        # 生成笔划坐标
        stk_coor0 = self.point_to_stk_coor(xy).permute(0, 2, 1)  # [bs, n_stk, emb]

        # 生成初始 sparse graph
        sparse_graph0 = self.point_to_sparse(xy)  # [bs, emb, n_stk]
        assert sparse_graph0.size()[2] == self.n_stk

        # 生成初始 dense graph
        dense_graph0 = self.point_to_dense(xy)
        assert dense_graph0.size()[2] == n_stk and dense_graph0.size()[3] == n_stk_pnt

        # 交叉更新数据
        sparse_graph1, dense_graph1, stk_coor1 = self.sd1(sparse_graph0, dense_graph0, stk_coor=stk_coor0)
        sparse_graph2, dense_graph2, stk_coor2 = self.sd2(sparse_graph1, dense_graph1, stk_coor=stk_coor1)

        # 提取全局特征
        sparse_glo0 = sparse_graph0.max(2)[0]
        sparse_glo1 = sparse_graph1.max(2)[0]
        sparse_glo2 = sparse_graph2.max(2)[0]

        dense_glo0 = dense_graph0.amax((2, 3))
        dense_glo1 = dense_graph1.amax((2, 3))
        dense_glo2 = dense_graph2.amax((2, 3))

        all_fea = torch.cat([sparse_glo0, sparse_glo1, sparse_glo2, dense_glo0, dense_glo1, dense_glo2], dim=1)

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
    atensor = torch.rand([bs, global_defs.n_stk, global_defs.n_stk_pnt, 2])

    classifier_cls = SDGraphCls(10, 2)
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





