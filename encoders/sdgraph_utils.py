import os.path
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from pathlib import Path

import encoders.utils as eu


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
            eu.activate_func(),
            nn.Linear(channel_time // 2, channel_time)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class TimeMerge(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            eu.activate_func(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim_in, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        """
        将输入的 tensor 融合时间特征
        :param x: [bs, emb, n_stk] or [bs, emb, n_stk, n_stk_pnt]
        :param time_emb:
        :return: [bs, emb, n_stk] or [bs, emb, n_stk, n_stk_pnt]
        """
        is_stk_stk_pnt = False

        if len(x.size()) == 4:
            is_stk_stk_pnt = True
            bs, emb, n_stk, n_stk_pnt = x.size()
            x = x.view(bs, emb, n_stk * n_stk_pnt)

        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        x_embd = h + self.res_conv(x)

        if is_stk_stk_pnt:
            x_embd = x_embd.view(bs, x_embd.size(1), n_stk, n_stk_pnt)

        return x_embd


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
# -----------------------------------------------------------------------------------------


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


class PointToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph'
    使用一维卷积及最大池化方式
    """
    def __init__(self, point_dim, sparse_out, with_time=False, time_emb_dim=0, dropout=0.0):
        super().__init__()

        self.with_time = with_time

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

        if self.with_time:
            self.time_merge = TimeMerge(sparse_out, sparse_out, time_emb_dim)

    def forward(self, xy, time_emb=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = xy.permute(0, 3, 1, 2)

        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = self.point_increase(xy)

        # -> [bs, emb, n_stk]
        xy = torch.max(xy, dim=3)[0]
        # assert xy.size(2) == self.n_stk

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            xy = self.time_merge(xy, time_emb)

        return xy


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
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        bs, n_stk, n_stk_pnt, channel = xy.size()

        xy = einops.rearrange(xy, 'b s sp c -> b c (s sp)')
        dense_emb = self.encoder(xy)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            dense_emb = self.time_merge(dense_emb, time_emb)

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


if __name__ == '__main__':
    c_root = Path(__file__).resolve()
    parent_dir = c_root.parent.parent
    save_root = os.path.join(parent_dir, 'imgs_gen', '1.png')

    print(save_root)



