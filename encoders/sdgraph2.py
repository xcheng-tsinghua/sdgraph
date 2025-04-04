import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils import full_connected, full_connected_conv1d, full_connected_conv2d, activate_func, fps, index_points, square_distance
import math
from einops import rearrange

import global_defs
from encoders.PointBERT_ULIP2 import create_pretrained_pointbert


# PRE_TRAINED_POINTBERT = create_pretrained_pointbert().cuda()


# def get_stk_coor(xy, n_stk, n_stk_pnt):
#     """
#     获取每个笔划的坐标
#     :param xy: [bs, 2, n_skh_pnt]
#     :param n_stk:
#     :param n_stk_pnt:
#     :return: [bs, n_stk, emb]
#     """
#     bs, _, n_skh_pnt = xy.size()
#     assert n_skh_pnt == n_stk * n_stk_pnt
#
#     xy = xy.reshape(bs * n_stk, n_stk_pnt, 2)
#     zeros = torch.zeros(bs * n_stk, n_stk_pnt, 1, device=xy.device, dtype=xy.dtype)
#     xy = torch.cat([xy, zeros], dim=2)
#
#     # pretrained_pointbert = create_pretrained_pointbert().to(xy.device)
#
#     xy = PRE_TRAINED_POINTBERT(xy)
#     xy = xy.view(bs, n_stk, PRE_TRAINED_POINTBERT.channel_out)
#
#     return xy


def knn(x, k):
    """
    找到最近的点的索引，包含自身
    :param x: [bs, 2, n_point]
    :param k:
    :return: [batch_size, num_points, k]
    """
    # -> x: [bs, 2, n_point]

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k+1)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # -> x: [bs, 2, n_point]

    batch_size, channel, num_points = x.size()

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class GCNEncoder(nn.Module):
    """
    实际上是 DGCNN Encoder
    """
    def __init__(self, emb_in, emb_out, n_near=10,):
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

        self.conv1 = full_connected_conv2d([emb_l1_0, emb_l1_1, emb_l1_2],
                                           final_proc=True,
                                           drop_rate=0.0
                                           )
        self.conv2 = full_connected_conv2d([emb_l2_0, emb_l2_1, emb_l2_2],
                                           final_proc=True,
                                           drop_rate=0.0
                                           )

        self.conv3 = full_connected_conv1d([emb_l3_0, emb_l3_1, emb_l3_2],
                                           final_proc=True, drop_rate=0.0
                                           )

    def forward(self, x):
        # x: [bs, channel, n_token]

        # -> [bs, emb, n_token, n_neighbor]
        x = get_graph_feature(x, k=self.n_near)
        x = self.conv1(x)

        # -> [bs, emb, n_token]
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_near)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # -> [bs, emb, n_token]
        x = torch.cat((x1, x2), dim=1)

        # -> [bs, emb, n_token]
        x = self.conv3(x)

        return x


class DownSample(nn.Module):
    """
    对sdgraph同时进行下采样
    该模块处理后笔划数和笔划中的点数同时降低为原来的1/2
    """
    def __init__(self, sp_in, sp_out, dn_in, dn_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        # 笔划数及单个笔划内的点数
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # sparse graph 特征更新层
        # self.sp_conv = full_connected_conv1d([sp_in, sp_out], True, dropout, True)

        # dense graph 下采样及特征更新
        self.dn_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dn_in,  # 输入通道数 (RGB)
                out_channels=dn_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dn_out),
            activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, sparse_fea, dense_fea, stk_coor, stk_fea_bef=None, n_sp_up_near=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_pnt]
        :param stk_coor: [bs, n_stk, emb] 每个笔划的原始特征，用于进行采样
        :param stk_fea_bef: 占位，因为上采样时和在采样时的输入不同
        :param n_sp_up_near: 占位
        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        n_stk = sparse_fea.size(2)
        assert n_stk == self.n_stk

        # --- 对sgraph进行下采样 ---
        # 使用FPS采样，获得采样得到的fps索引
        # stk_coor = stk_coor.permute(0, 2, 1)  # -> [bs, n_stk, emb]
        fps_idx = fps(stk_coor, self.n_stk // 2)  # -> [bs, n_stk // 2]

        # 根据索引获得对应的笔划初始特征
        stk_coor_sampled = index_points(stk_coor, fps_idx)  # -> [bs, n_stk // 2, emb]
        assert stk_coor_sampled.size(1) == self.n_stk // 2

        # 根据索引获得对应的sgraph特征
        sparse_fea = index_points(sparse_fea.permute(0, 2, 1), fps_idx).permute(0, 2, 1)  # -> [bs, emb, n_stk // 2]
        assert sparse_fea.size(2) == self.n_stk // 2

        # --- 对dense fea进行下采样 ---
        # 先找到对应的笔划
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt).permute(0, 2, 3, 1)  # -> [bs, n_stk, n_stk_pnt, emb]
        dense_fea = dense_fea.reshape(bs, self.n_stk, self.n_stk_pnt * emb)  # -> [bs, n_stk, n_stk_pnt * emb]
        dense_fea = index_points(dense_fea, fps_idx)  # -> [bs, n_stk // 2, n_stk_pnt * emb]
        dense_fea = dense_fea.view(bs, self.n_stk // 2, self.n_stk_pnt, emb)  # -> [bs, n_stk // 2, n_stk_pnt, emb]

        # 进行下采样
        dense_fea = dense_fea.permute(0, 3, 1, 2)  # -> [bs, emb, n_stk // 2, n_stk_pnt]
        dense_fea = self.dn_conv(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt // 2]
        dense_fea = dense_fea.view(bs, dense_fea.size(1), (self.n_stk // 2) * (self.n_stk_pnt // 2))  # -> [bs, emb, n_stk * n_stk_pnt // 4]

        return sparse_fea, dense_fea, stk_coor_sampled



        #
        # # 然后找到特征空间中与之最近的笔划进行maxPooling
        # # 找到每个笔划附近的3个笔划的索引，包含自身
        # knn_idx = knn(stk_fea.permute(0, 2, 1), 2)  # -> [bs, n_stk, 2]
        # assert knn_idx.size(2) == 2
        #
        # # 找到fps得到的中心点的对应的附近点的索引
        # knn_idx = index_points(knn_idx, fps_idx)  # -> [bs, n_stk // 2, 2]
        # assert knn_idx.size(1) == self.n_stk // 2
        #
        # # 找到每个点附近的2个点的特征
        # sparse_fea = index_points(sparse_fea.permute(0, 2, 1), knn_idx).permute(0, 3, 1, 2)  # -> [bs, emb, n_stk // 2, 2]
        # assert sparse_fea.size(2) == self.n_stk // 2 and sparse_fea.size(3) == 2
        #
        # # 最大池化得到中心点的特征
        # sparse_fea = sparse_fea.max(3)[0]  # -> [bs, emb, n_stk // 2]
        # assert sparse_fea.size(2) == self.n_stk // 2
        #
        # # 更新sgraph的特征
        # sparse_fea = self.sp_conv(sparse_fea)
        #
        # # 获取采样后的笔划特征
        # stk_fea_sampled = index_points(stk_fea, fps_idx).permute(0, 2, 1)  # -> [bs, emb, n_stk // 2]
        # assert stk_fea_sampled.size(2) == self.n_stk // 2
        #
        # # --- 对dense fea进行下采样 ---
        # # 先找到对应的笔划
        # dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        # dense_fea = dense_fea.permute(0, 2, 3, 1)  # -> [bs, n_stk, n_stk_pnt, emb]
        # dense_fea = dense_fea.reshape(bs, self.n_stk, self.n_stk_pnt * emb)  # -> [bs, n_stk, n_stk_pnt * emb]
        # dense_fea = index_points(dense_fea, fps_idx)  # -> [bs, n_stk // 2, n_stk_pnt * emb]
        # dense_fea = dense_fea.view(bs, self.n_stk // 2, self.n_stk_pnt, emb)  # -> [bs, n_stk // 2, n_stk_pnt, emb]
        #
        # # 进行下采样
        # dense_fea = dense_fea.permute(0, 3, 1, 2)  # -> [bs, emb, n_stk // 2, n_stk_pnt]
        # dense_fea = self.dn_conv(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt // 2]
        # dense_fea_emb = dense_fea.size(1)
        # dense_fea = dense_fea.view(bs, dense_fea_emb, (self.n_stk // 2) * (self.n_stk_pnt // 2))  # -> [bs, emb, n_stk * n_stk_pnt // 4]
        #
        # return sparse_fea, dense_fea, stk_fea_sampled


class UpSample(nn.Module):
    def __init__(self, sp_in, sp_out, dn_in, dn_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.sp_conv = full_connected_conv1d([sp_in, sp_out], True, dropout, True)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dn_in,  # 输入通道数
                out_channels=dn_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dn_out),
            activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, sparse_fea, dense_fea, stk_fea, stk_fea_bef, n_sp_up_near=3):
        """
        对于 stk_fea_bef 中的某个点(center)，找到 stk_fea 中与之最近的 n_sp_up_near 个点(nears)，将nears的特征进行加权和，得到center的插值特征
        nears中第i个点(near_i)特征的权重为 [1/d(near_i)]/sum(k=1->n_sp_up_near)[1/d(near_k)]
        d(near_i)为 center到near_i的距离，即距离越近，权重越大
        之后拼接 sparse_fea_bef 与插值后的 sparse_fea，再利用MLP对每个点的特征单独进行处理

        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_pnt]
        :param stk_fea: 采样后的笔划特征 [bs, emb, n_stk]
        :param stk_fea_bef: 采样前的笔划特征 [bs, emb, n_stk * 2]
        :param n_sp_up_near: sgraph上采样过程中搜寻的附近点点数

        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        # 对sgraph进行上采样
        # 计算sparse_fea_bef中的每个点到sparse_fea中每个点的距离 sparse_fea_bef:[bs, emb, n_stk * 2], sparse_fea:[bs, emb, n_stk], return: [bs, n_stk * 2, n_stk]
        dists = square_distance(stk_fea_bef.permute(0, 2, 1), stk_fea.permute(0, 2, 1))
        assert dists.size(1) == self.n_stk * 2 and dists.size(2) == self.n_stk

        # 计算每个初始点到采样点距离最近的3个点，sort 默认升序排列, 取三个
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :n_sp_up_near], idx[:, :, :n_sp_up_near]  # [B, N, 3]

        # 最近距离的每行求倒数
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)

        # 求倒数后每行中每个数除以该行之和
        weight = dist_recip / norm  # ->[B, N, n_sp_up_near]

        # index_points(points2, idx): 为原始点集中的每个点找到采样点集中与之最近的3个三个点的特征 -> [B, N, 3, D]
        sparse_fea = index_points(sparse_fea.permute(0, 2, 1), idx)  # -> [bs, n_stk * 2, 3, emb]
        weight = weight.view(bs, self.n_stk * 2, n_sp_up_near, 1)  # -> [bs, n_stk * 2, 3, 1]
        sparse_fea = torch.sum(sparse_fea * weight, dim=2)  # -> [bs, n_stk * 2, emb]
        sparse_fea = sparse_fea.permute(0, 2, 1)  # -> [bs, emb, n_stk * 2]
        assert sparse_fea.size(2) == self.n_stk * 2
        sparse_fea = self.sp_conv(sparse_fea)

        # 为dense graph进行笔划层级上采样
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        dense_fea_emb = dense_fea.size(1)
        dense_fea = dense_fea.permute(0, 2, 3, 1)  # -> [bs, n_stk, n_stk_pnt, emb]
        dense_fea = dense_fea.reshape(bs, self.n_stk, self.n_stk_pnt * dense_fea_emb)  # -> [bs, n_stk, n_stk_pnt * emb]
        dense_fea = index_points(dense_fea, idx)  # -> [bs, n_stk * 2, 3, n_stk_pnt * emb]
        dense_fea = torch.sum(dense_fea * weight, dim=2)  # -> [bs, n_stk * 2, n_stk_pnt * emb]
        dense_fea = dense_fea.view(bs, self.n_stk * 2, self.n_stk_pnt, dense_fea_emb)
        dense_fea = dense_fea.permute(0, 3, 1, 2)  # -> [bs, emb, n_stk * 2, n_stk_pnt]

        dense_fea = self.conv(dense_fea)  # -> [bs, emb, n_stk * 2, n_stk_pnt * 2]
        dense_fea = dense_fea.view(bs, dense_fea.size(1), (self.n_stk * self.n_stk_pnt) * 4)

        return sparse_fea, dense_fea, None


class PointToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph'
    使用一维卷积及最大池化方式
    """
    def __init__(self, point_dim, sparse_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt,
                 with_time=False, time_emb_dim=0):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        mid_dim = int((point_dim * sparse_out) ** 0.5)
        self.point_increase = nn.Sequential(
            nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
            nn.BatchNorm2d(mid_dim),
            activate_func(),
            # nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
            nn.BatchNorm2d(sparse_out),
            activate_func(),
            # nn.Dropout2d(dropout),
        )

        if self.with_time:
            self.time_merge = TimeMerge(sparse_out, sparse_out, time_emb_dim)

    def forward(self, xy, time_emb=None):
        """
        :param xy: [bs, emb, n_stk * n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk]
        """
        bs, emb, _ = xy.size()

        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = xy.view(bs, emb, self.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = self.point_increase(xy)

        # -> [bs, emb, n_stk]
        xy = torch.max(xy, dim=3)[0]
        assert xy.size(2) == self.n_stk

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            xy = self.time_merge(xy, time_emb)

        return xy

#
# class PointToSparse(nn.Module):
#     """
#     将 dense graph 的数据转移到 sparse graph'
#     使用一维卷积及最大池化方式
#     """
#     def __init__(self,
#                  n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt,
#                  with_time=False, time_emb_dim=0
#                  ):
#         super().__init__()
#
#         self.n_stk = n_stk
#         self.n_stk_pnt = n_stk_pnt
#         self.with_time = with_time
#
#         # mid_dim = int((point_dim * sparse_out) ** 0.5)
#         # self.point_increase = nn.Sequential(
#         #     nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
#         #     nn.BatchNorm2d(mid_dim),
#         #     activate_func(),
#         #     # nn.Dropout2d(dropout),
#         #
#         #     nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
#         #     nn.BatchNorm2d(sparse_out),
#         #     activate_func(),
#         #     # nn.Dropout2d(dropout),
#         # )
#
#         if self.with_time:
#             self.time_merge = TimeMerge(self.pointbert.channel_out, self.pointbert.channel_out, time_emb_dim)
#
#     def forward(self, xy, time_emb=None):
#         """
#         :param xy: [bs, emb, n_stk * n_stk_pnt]
#         :param time_emb: [bs, emb]
#         :return: [bs, emb, n_stk]
#         """
#         bs, emb, _ = xy.size()
#
#         # -> [bs, emb, n_stk, n_stk_pnt]
#         xy = xy.view(bs, emb, self.n_stk, self.n_stk_pnt)
#
#         # -> [bs, n_stk, n_stk_pnt, emb]
#         xy = xy.permute(0, 2, 3, 1)
#
#         # otherxy = xy
#
#         xy = xy.reshape(bs * self.n_stk, self.n_stk_pnt, emb)
#         zeros = torch.zeros(xy.size(0), xy.size(1), 1, device=xy.device, dtype=xy.dtype)
#         xy = torch.cat([xy, zeros], dim=2)
#
#         xy = PRE_TRAINED_POINTBERT(xy)
#
#         xy = xy.view(bs, self.n_stk, PRE_TRAINED_POINTBERT.channel_out)
#
#         # other
#         # new_xy = []
#         # for i in range(self.n_stk):
#         #     c_other_xy = otherxy[:, i, :, :]  # -> [bs, n_stk_pnt, 2]
#         #     zeros2 = torch.zeros(c_other_xy.size(0), c_other_xy.size(1), 1, device=c_other_xy.device, dtype=c_other_xy.dtype)
#         #     c_other_xy = torch.cat([c_other_xy, zeros2], dim=2)
#         #     c_other_emb = PRE_TRAINED_POINTBERT(c_other_xy).unsqueeze(1)  # -> [bs, 1, emb]
#         #     new_xy.append(c_other_emb)
#         #
#         # new_xy = torch.cat(new_xy, dim=1)
#         #
#         # asasaaaa = new_xy - xy
#         #
#         # maxval = torch.max(asasaaaa)
#         # print(maxval)
#
#         xy = xy.permute(0, 2, 1)
#
#         assert xy.size(2) == self.n_stk
#
#         assert self.with_time ^ (time_emb is None)
#         if self.with_time:
#             xy = self.time_merge(xy, time_emb)
#
#         return xy


class PointToDense(nn.Module):
    """
    利用点坐标生成 dense graph
    使用DGCNN直接为每个点生成对应特征
    """
    def __init__(self, point_dim, emb_dim, with_time=False, time_emb_dim=0, n_near=10):
        super().__init__()
        self.encoder = GCNEncoder(point_dim, emb_dim, n_near)

        self.with_time = with_time
        if self.with_time:
            self.time_merge = TimeMerge(emb_dim, emb_dim, time_emb_dim)

    def forward(self, xy, time_emb=None):
        """
        :param xy: [bs, emb, n_stk * n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk * n_stk_pnt]
        """
        dense_emb = self.encoder(xy)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            dense_emb = self.time_merge(dense_emb, time_emb)

        return dense_emb


class SparseToDense(nn.Module):
    """
    将sgraph转移到dgraph
    直接拼接到该笔划对应的点
    """
    def __init__(self, n_stk, n_stk_pnt):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk * n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk * n_stk_pnt]
        """
        bs, emb, _ = dense_fea.size()

        # -> [bs, emb, n_stk, n_stk_pnt]
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)

        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)

        # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)

        # -> [bs, emb, n_stk * n_stk_pnt]
        union_dense = union_dense.view(bs, union_dense.size(1), self.n_stk * self.n_stk_pnt)

        return union_dense


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self, dense_in, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.dense_to_sparse = nn.Sequential(
            nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
            nn.BatchNorm2d(dense_in),
            activate_func(),
            nn.Dropout2d(dropout),
        )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk * n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        bs, emb, _ = dense_fea.size()

        # -> [bs, emb, n_stk, n_stk_pnt]
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk, n_stk_pnt]
        sparse_feas_from_dense = self.dense_to_sparse(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = torch.max(sparse_feas_from_dense, dim=3)[0]
        assert sparse_feas_from_dense.size(2) == self.n_stk

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)

        return union_sparse


class SinusoidalPosEmb(nn.Module):
    """
    将时间步t转化为embedding
    """
    def __init__(self, dim, theta):  # 256， 10000
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        :param x: 时间步 [bs, ]
        :return:
        """
        device = x.device
        half_dim = self.dim // 2  # 128
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEncode(nn.Module):
    """
    编码时间步
    """
    def __init__(self, channel_time):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(channel_time // 4, theta=10000),
            nn.Linear(channel_time // 4, channel_time // 2),
            activate_func(),
            nn.Linear(channel_time // 2, channel_time)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class TimeMerge(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            activate_func(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim_in, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """
        :param x: [bs, channel, n_node]
        :param scale_shift: [bs, ]
        :return:
        """
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.dropout(self.act(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        """
        :param dim: forward过程中输入x的特征长度
        """
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SDGraphEncoder(nn.Module):
    """
    包含笔划及笔划中的点层级的下采样与上采样
    """
    def __init__(self,
                 sparse_in, sparse_out, dense_in, dense_out,  # 输入输出维度
                 n_stk, n_stk_pnt,  # 笔划数，每个笔划中的点数
                 sp_near=10, dn_near=10,  # 更新sdgraph的两个GCN中邻近点数目
                 sample_type='down_sample',  # 采样类型
                 with_time=False, time_emb_dim=0,  # 是否附加时间步
                 dropout=0.2
                 ):
        """
        :param sample_type: [down_sample, up_sample, none]
        """
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        self.dense_to_sparse = DenseToSparse(dense_in, n_stk, n_stk_pnt, dropout)
        self.sparse_to_dense = SparseToDense(n_stk, n_stk_pnt)

        self.sparse_update = GCNEncoder(sparse_in + dense_in, sparse_out, sp_near)
        self.dense_update = GCNEncoder(dense_in + sparse_in, dense_out, dn_near)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)
            self.time_mlp_dn = TimeMerge(dense_out, dense_out, time_emb_dim, dropout)

    def forward(self, sparse_fea, dense_fea, time_emb=None, stk_coor=None, stk_coor_bef=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_point]
        :param stk_coor: 初始用点获取的每个笔划的特征
        :param stk_coor_bef: 采样前的笔划特征
        :param time_emb: [bs, emb]
        :return:
        """
        # 确保在上采样时stk_coor_bef不为None，且下采样时stk_coor_bef为None， ^ :异或，两者不同为真
        assert (self.sample_type == 'up_sample') ^ (stk_coor_bef is None)

        bs, emb, n_stk = sparse_fea.size()
        assert n_stk == self.n_stk

        n_points = dense_fea.size()[2]
        assert n_points == self.n_stk * self.n_stk_pnt

        # 信息交换
        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)
        union_dense = self.sparse_to_dense(sparse_fea, dense_fea)

        # 信息更新
        union_sparse = self.sparse_update(union_sparse)
        union_dense = self.dense_update(union_dense)

        # 采样
        union_sparse, union_dense, stk_coor_sampled = self.sample(union_sparse, union_dense, stk_coor, stk_coor_bef, 3)
        if stk_coor_sampled is not None:
            assert stk_coor_sampled.size(0) == bs and (stk_coor_sampled.size(1) == self.n_stk // 2 or stk_coor_sampled.size(1) == self.n_stk * 2)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense, stk_coor_sampled


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, n_stk: int = global_defs.n_stk, n_stk_pnt: int = global_defs.n_stk_pnt, dropout: float = 0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('cls 双下采样, use stk coor')

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # 各层特征维度
        sparse_l0 = 32 + 16 + 8
        sparse_l1 = 128 + 64 + 32
        sparse_l2 = 512 + 256 + 128

        dense_l0 = 32 + 16
        dense_l1 = 128 + 64
        dense_l2 = 512 + 256

        # 生成笔划坐标（特征）
        # self.point_bert = create_pretrained_pointbert().cuda()

        # 生成初始 sdgraph
        self.point_to_sparse = PointToSparse(2, sparse_l0)
        self.point_to_dense = PointToDense(2, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt,
                                  dropout=dropout
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk // 2, n_stk_pnt=self.n_stk_pnt // 2,
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

        self.linear = full_connected(channels=[out_l0, out_l1, out_l2, out_l3], final_proc=False, drop_rate=dropout)

    # def get_stk_coor(self, xy):
    #     """
    #     获取每个笔划的坐标
    #     :param xy: [bs, 2, n_skh_pnt]
    #     :param n_stk:
    #     :param n_stk_pnt:
    #     :return: [bs, n_stk, emb]
    #     """
    #     xy = xy.permute(0, 2, 1)
    #     bs, n_skh_pnt, pnt_channel = xy.size()
    #     assert n_skh_pnt == self.n_stk * self.n_stk_pnt and pnt_channel == 2
    #
    #     xy = xy.reshape(bs * self.n_stk, self.n_stk_pnt, 2)
    #     zeros = torch.zeros(bs * self.n_stk, self.n_stk_pnt, 1, device=xy.device, dtype=xy.dtype)
    #     xy = torch.cat([xy, zeros], dim=2)
    #
    #     xy = self.point_bert(xy)
    #     xy = xy.view(bs, self.n_stk, self.point_bert.channel_out)
    #
    #     return xy

    def forward(self, xy):
        """
        :param xy: [bs, 2, n_skh_pnt]
        :param stk_coor: [bs, n_stk, 512]  笔划坐标，由pointbert_ulip2生成
        :return: [bs, n_classes]
        """
        xy = xy[:, :2, :]

        bs, channel, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt and channel == 2

        # 生成初始 sparse graph
        sparse_graph0 = self.point_to_sparse(xy)
        assert sparse_graph0.size()[2] == self.n_stk

        # 生成初始 dense graph
        dense_graph0 = self.point_to_dense(xy)
        assert dense_graph0.size()[2] == n_point

        # 获取笔划坐标
        stk_coor0 = dense_graph0
        # stk_coor = xy.view(bs, 2, self.n_stk, self.n_stk_pnt)
        # stk_coor = stk_coor.permute(0, 2, 3, 1)
        # stk_coor = stk_coor.reshape(bs, self.n_stk, self.n_stk_pnt * 2)

        # 交叉更新数据
        sparse_graph1, dense_graph1, stk_coor1 = self.sd1(sparse_graph0, dense_graph0, stk_coor=stk_coor0)
        sparse_graph2, dense_graph2, stk_coor2 = self.sd2(sparse_graph1, dense_graph1, stk_coor=stk_coor1)

        # 提取全局特征
        sparse_glo0 = sparse_graph0.max(2)[0]
        sparse_glo1 = sparse_graph1.max(2)[0]
        sparse_glo2 = sparse_graph2.max(2)[0]

        dense_glo0 = dense_graph0.max(2)[0]
        dense_glo1 = dense_graph1.max(2)[0]
        dense_glo2 = dense_graph2.max(2)[0]

        all_fea = torch.cat([sparse_glo0, sparse_glo1, sparse_glo2, dense_glo0, dense_glo1, dense_glo2], dim=1)

        # 利用全局特征分类
        cls = self.linear(all_fea)
        cls = F.log_softmax(cls, dim=1)

        return cls


class SDGraphUNet(nn.Module):
    def __init__(self, channel_in, channel_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.0):
        super().__init__()
        print('diff 笔划下采样')

        '''草图参数'''
        self.channel_in = channel_in
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        '''各层通道数'''
        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        global_dim = 1024
        time_emb_dim = 256

        '''时间步特征生成层'''
        self.time_encode = TimeEncode(time_emb_dim)

        '''点坐标 -> 初始 sdgraph 生成层'''
        self.point_to_sparse = PointToSparse(channel_in, sparse_l0, with_time=True, time_emb_dim=time_emb_dim)
        self.point_to_dense = PointToDense(channel_in, dense_l0, with_time=True, time_emb_dim=time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                       self.n_stk, self.n_stk_pnt,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                       self.n_stk // 2, self.n_stk_pnt // 2,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        '''全局特征生成层'''
        global_in = sparse_l2 + dense_l2
        self.global_linear = full_connected(
            channels=[global_in, int((global_in * global_dim) ** 0.5), global_dim],
            final_proc=True, drop_rate=dropout)

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     n_stk=self.n_stk // 4, n_stk_pnt=self.n_stk_pnt // 4,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     n_stk=self.n_stk // 2, n_stk_pnt=self.n_stk_pnt // 2,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        '''最终输出层'''
        final_in = dense_l0 + sparse_l0 + dense_l1 + sparse_l1 + channel_in
        self.final_linear = full_connected_conv1d(
            channels=[final_in, int((channel_out * final_in) ** 0.5), channel_out],
            final_proc=False,
            drop_rate=dropout
        )

    def channels(self):
        return self.channel_in

    def forward(self, xy, time):
        """
        :param xy: [bs, channel_in, n_skh_pnt]
        :param time: [bs, ]
        :return: [bs, channel_out, n_skh_pnt]
        """
        '''生成时间步特征'''
        time_emb = self.time_encode(time)

        bs, channel_in, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt and channel_in == self.channel_in

        '''生成初始 sdgraph'''
        sparse_graph_up0 = self.point_to_sparse(xy, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_up0 = self.point_to_dense(xy, time_emb)  # -> [bs, emb, n_point]
        assert sparse_graph_up0.size()[2] == self.n_stk and dense_graph_up0.size()[2] == n_point

        stk_fea_l0 = sparse_graph_up0

        '''下采样'''
        sparse_graph_up1, dense_graph_up1, stk_fea_l1 = self.sd_down1(sparse_graph_up0, dense_graph_up0, time_emb, stk_fea_l0)
        sparse_graph_up2, dense_graph_up2, stk_fea_l2 = self.sd_down2(sparse_graph_up1, dense_graph_up1, time_emb, stk_fea_l1)

        '''获取全局特征'''
        sp_up2_glo = sparse_graph_up2.max(2)[0]
        dn_up2_glo = dense_graph_up2.max(2)[0]

        fea_global = torch.cat([sp_up2_glo, dn_up2_glo], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征 (直接拼接在后面)'''
        sparse_fit = fea_global.unsqueeze(2).repeat(1, 1, sparse_graph_up2.size(2))
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = fea_global.unsqueeze(2).repeat(1, 1, dense_graph_up2.size(2))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1, _ = self.sd_up2(sparse_graph_down2, dense_graph_down2, time_emb, stk_fea_l2, stk_fea_l1)  # -> [bs, sp_l2, n_stk], [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0, _ = self.sd_up1(sparse_graph_down1, dense_graph_down1, time_emb, stk_fea_l1, stk_fea_l0)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = sparse_graph.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel_in, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)
        return noise



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
    atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
    t1 = torch.randint(0, 1000, (bs,)).long().cuda()

    # classifier = SDGraphUNet(2, 2).cuda()
    # cls11 = classifier(atensor, t1)

    classifier = SDGraphCls(10).cuda()
    cls11 = classifier(atensor)

    print(cls11.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





