import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from encoders.Dgcnn import DgcnnEncoder
from encoders.utils import full_connected, full_connected_conv1d, full_connected_conv2d
import global_defs


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


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """

        :param x: [bs, channel, n_node]
        :param scale_shift: [bs, ]
        :return:
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.dropout(self.act(x))
        return x


class TimeMerge(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
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


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    """
    def __init__(self, dense_dim, sparse_dim, dropout=0.4):
        super().__init__()

        # 将 DGraph 的数据转移到 SGraph
        ps_mid = int((dense_dim * sparse_dim) ** 0.5)
        self.dense_to_sparse = nn.Sequential(
            nn.Conv2d(in_channels=dense_dim, out_channels=ps_mid, kernel_size=(1, 2)),
            nn.BatchNorm2d(ps_mid),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=ps_mid, out_channels=sparse_dim, kernel_size=(1, 2)),
            nn.BatchNorm2d(sparse_dim),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return:
        """
        # -> [bs, emb, n_stk, n_stk_pnt-2]
        sparse_feas_from_dense = self.dense_to_sparse(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = torch.max(sparse_feas_from_dense, dim=3)[0]
        return sparse_feas_from_dense


class DenseToSparseAttn(nn.Module):
    """
    使用注意力机制实现将 dgraph 的特征转移到 sgraph
    使用向量注意力机制
    :return:
    """
    def __init__(self, emb_sp, emb_dn, emb_final, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk_pnt = n_stk_pnt

        self.sp_q = full_connected_conv2d([emb_sp, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.dn_k = full_connected_conv2d([emb_dn, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.sp_v = full_connected_conv2d([emb_sp, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.gamma = full_connected_conv2d([emb_final, emb_final], bias=False, drop_rate=dropout, final_proc=True)

    def forward(self, sparse_fea, dense_fea):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk]
        """
        bs, emb, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == global_defs.n_stk and n_stk_pnt == self.n_stk_pnt

        # -> [bs, emb, n_stk, 1]
        sparse_fea = sparse_fea.unsqueeze(3)
        assert sparse_fea.size(2) == global_defs.n_stk

        sp_q = self.sp_q(sparse_fea)  # -> [bs, emb, n_stk, 1]
        dn_k = self.dn_k(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = self.sp_v(sparse_fea)  # -> [bs, emb, n_stk, 1]

        coef = F.softmax(self.gamma(sp_q - dn_k), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = coef * sp_v  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = torch.sum(sp_v, dim=3)  # -> [bs, emb, n_stk]

        return sp_v


class SparseToDenseAttn(nn.Module):
    """
    使用注意力机制实现将 sgraph 的特征转移到 dgraph
    使用线性注意力机制
    :return:
    """
    def __init__(self, emb_sp, emb_dn, emb_final, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.emb_final = emb_final
        self.n_stk_pnt = n_stk_pnt

        self.dn_q = full_connected_conv2d([emb_dn, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.sp_k = full_connected_conv2d([emb_sp, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.dn_v = full_connected_conv2d([emb_dn, emb_final], bias=False, drop_rate=dropout, final_proc=True)
        self.gamma = full_connected_conv2d([emb_final, emb_final], bias=False, drop_rate=dropout, final_proc=True)

    def forward(self, sparse_fea, dense_fea):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        bs, emb, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == global_defs.n_stk and n_stk_pnt == self.n_stk_pnt

        # -> [bs, emb, n_stk, 1]
        sparse_fea = sparse_fea.unsqueeze(3)
        assert sparse_fea.size(2) == global_defs.n_stk

        dn_q = self.dn_q(dense_fea)  # -> [bs, emb, n_stk, 1]
        sp_k = self.sp_k(sparse_fea)  # -> [bs, emb, n_stk, n_stk_pnt]
        dn_v = self.dn_v(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt]

        # y_i = emb * F.softmax(self.gamma(fai_xi - psi_xj), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        coef = F.softmax(self.gamma(dn_q - sp_k), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        dn_v = coef * dn_v  # -> [bs, emb, n_stk, n_stk_pnt]

        dn_v = dn_v.view(bs, self.emb_final, n_stk * n_stk_pnt)

        return dn_v


def down_sample(dim_in, dim_out, dropout=0.4):
    """
    将 dense graph 的每个笔划的点数调整为原来的 1/2
    :param dim_in:
    :param dim_out:
    :param dropout:
    :return:
    """
    # return nn.Sequential(
    #     Rearrange('b c s (p d1) -> b (c d1) s p', d1=2),
    #     nn.Conv2d(dim_in * 2, dim_out, 1),
    #     nn.BatchNorm2d(dim_out),
    #     nn.GELU(),
    #     nn.Dropout2d(dropout)
    # )

    return nn.Sequential(
        nn.Conv2d(
            in_channels=dim_in,  # 输入通道数 (RGB)
            out_channels=dim_out,  # 输出通道数 (保持通道数不变)
            kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
            stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
            padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
        ),
        nn.BatchNorm2d(dim_out),
        nn.GELU(),
        nn.Dropout2d(dropout)
    )


def up_sample(dim_in, dim_out, dropout=0.4):
    """
    将 dense graph 的每个笔划的点数调整为原来的 2 倍
    :param dim_in:
    :param dim_out:
    :param dropout:
    :return:
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=dim_in,  # 输入通道数
            out_channels=dim_out,  # 输出通道数
            kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
            stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
            padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
        ),
        nn.BatchNorm2d(dim_out),
        nn.GELU(),
        nn.Dropout2d(dropout)
    )


class SDGraphEncoder(nn.Module):
    def __init__(self, sparse_in, sparse_out, dense_in, dense_out, n_stk, n_stk_pnt, sample_type='down_sample'):
        """
        :param sparse_in:
        :param sparse_out:
        :param dense_in:
        :param dense_out:
        :param n_stk:
        :param n_stk_pnt:
        :param sample_type: [down_sample, up_sample, none]
        """
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # 将 DGraph 的数据转移到 SGraph
        # self.dense_to_sparse = DenseToSparse(dense_in, dense_in)

        self.dense_to_sparse = DenseToSparseAttn(sparse_in, dense_in, dense_in + sparse_in, n_stk_pnt)

        self.sparse_update = DgcnnEncoder(dense_in + sparse_in, sparse_out, n_near=2)
        self.dense_update = DgcnnEncoder(dense_in + sparse_in, dense_out, n_near=10)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = down_sample(dense_out, dense_out)
        elif self.sample_type == 'up_sample':
            self.sample = up_sample(dense_out, dense_out)
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

    def forward(self, sparse_fea, dense_fea):
        """
        :param sparse_fea: [bs, emb, n_stroke]
        :param dense_fea: [bs, emb, n_point]
        :return:
        """
        bs, _, n_stroke = sparse_fea.size()
        assert n_stroke == self.n_stk

        _, emb_dn, n_points = dense_fea.size()
        assert n_points == self.n_stk * self.n_stk_pnt

        # -> [bs, emb, n_stroke, stroke_point]
        dense_fea = dense_fea.view(bs, emb_dn, self.n_stk, self.n_stk_pnt)

        # 将 dense fea更新到 sparse graph
        # sparse_feas_from_dense = self.dense_to_sparse(dense_fea)

        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)

        # -> [bs, emb, n_stroke]
        # union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)
        # assert union_sparse.size()[2] == self.n_stk

        # 将 sparse fea更新到 dense graph
        # -> [bs, emb, n_stroke, stroke_point]
        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        assert dense_feas_from_sparse.size()[2] == self.n_stk and dense_feas_from_sparse.size()[3] == self.n_stk_pnt

        # -> [bs, emb, n_stroke, stroke_point]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)
        assert union_dense.size()[2] == self.n_stk and union_dense.size()[3] == self.n_stk_pnt

        # -> [bs, emb, n_stroke * stroke_point]
        union_dense = union_dense.view(bs, -1, self.n_stk * self.n_stk_pnt)

        # update sparse fea
        union_sparse = self.sparse_update(union_sparse)

        # update dense fea
        union_dense = self.dense_update(union_dense)

        union_dense = union_dense.view(bs, union_dense.size(1), self.n_stk, self.n_stk_pnt)
        union_dense = self.sample(union_dense)

        if self.sample_type == 'down_sample':
            union_dense = union_dense.view(bs, union_dense.size(1), (self.n_stk * self.n_stk_pnt) // 2)
        elif self.sample_type == 'up_sample':
            union_dense = union_dense.view(bs, union_dense.size(1), (self.n_stk * self.n_stk_pnt) * 2)
        elif self.sample_type == 'none':
            union_dense = union_dense.view(bs, union_dense.size(1), self.n_stk * self.n_stk_pnt)
        else:
            raise ValueError('invalid sample type')

        return union_sparse, union_dense


class SDGraph(nn.Module):
    def __init__(self, n_class: int):
        """
        :param n_class: 总类别数
        """
        super().__init__()

        self.channels = 2
        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt
        self.n_pnts = self.n_stk * self.n_stk_pnt

        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        # 将点坐标生成 sdgraph 的初步特征
        self.point_to_sparse = DenseToSparse(2, sparse_l0)
        self.point_to_dense = DgcnnEncoder(2, dense_l0)

        # 下采样1层
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt, sample_type='down_sample')

        # 下采样2层
        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2, sample_type='down_sample')

        outlayer_l0 = sparse_l2 + dense_l2
        outlayer_l1 = int(((sparse_l2 + dense_l2) * n_class) ** 0.5)
        outlayer_l2 = n_class

        self.linear = full_connected(channels=[outlayer_l0, outlayer_l1, outlayer_l2], final_proc=False)

    def forward(self, xy):
        # -> [bs, 2, n_point]
        xy = xy[:, :2, :]

        bs, channel, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt
        assert channel == 2

        # -> [bs, channel, n_stroke, stroke_point]
        xy = xy.view(bs, channel, self.n_stk, self.n_stk_pnt)

        # 生成 sparse graph
        # -> [bs, emb, n_stk]
        sparse_graph_up0 = self.point_to_sparse(xy)
        assert sparse_graph_up0.size()[2] == self.n_stk

        # -> [bs, 2, n_point]
        xy = xy.view(bs, channel, n_point)
        assert xy.size()[1] == 2

        # 生成 dense graph
        # -> [bs, emb, n_point]
        dense_graph_up0 = self.point_to_dense(xy)
        assert dense_graph_up0.size()[2] == n_point

        sparse_graph_up1, dense_graph_up1 = self.sd1(sparse_graph_up0, dense_graph_up0)
        sparse_graph_up2, dense_graph_up2 = self.sd2(sparse_graph_up1, dense_graph_up1)

        sparse_global = torch.max(sparse_graph_up2, dim=2)[0]
        dense_global = torch.max(dense_graph_up2, dim=2)[0]

        fea_global = torch.cat([sparse_global, dense_global], dim=1)

        cls = self.linear(fea_global)
        cls = F.log_softmax(cls, dim=1)

        return cls


class SDGraphUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 2
        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt
        self.n_pnts = self.n_stk * self.n_stk_pnt

        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        global_dim = 1024
        time_emb_dim = 256

        # --- 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 4, theta=10000),
            nn.Linear(time_emb_dim // 4, time_emb_dim // 2),
            nn.GELU(),
            nn.Linear(time_emb_dim // 2, time_emb_dim)
        )

        # 将点坐标生成 sdgraph 的初步特征
        self.point_to_sparse = DenseToSparse(2, sparse_l0)
        self.point_to_dense = DgcnnEncoder(2, dense_l0)
        self.tm_sp_l0 = TimeMerge(sparse_l0, sparse_l0, time_emb_dim)
        self.tm_dn_l0 = TimeMerge(dense_l0, dense_l0, time_emb_dim)

        # 下采样1层
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt, sample_type='down_sample')
        self.tm_sp_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        # 下采样2层
        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2, sample_type='down_sample')
        self.tm_sp_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        # 获取全局信息
        self.global_linear = full_connected(
            channels=[sparse_l2 + dense_l2, int(((sparse_l2 + dense_l2) * global_dim) ** 0.5), global_dim],
            final_proc=True)

        # 上采样2层
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 4, sample_type='up_sample')
        self.tm_sp_dec_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_dec_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        # 上采样1层
        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2, sample_type='up_sample')
        self.tm_sp_dec_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_dec_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        # 特征混合层
        self.sd_up0 = SDGraphEncoder(sparse_l1 + sparse_l0, sparse_l0,
                                     dense_l1 + dense_l0, dense_l0,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt, sample_type='none')
        self.tm_sp_dec_l0 = TimeMerge(sparse_l0, sparse_l0, time_emb_dim)
        self.tm_dn_dec_l0 = TimeMerge(dense_l0, dense_l0, time_emb_dim)

        # 输出层
        final_in = dense_l0 + sparse_l0 + 2
        self.final_linear = full_connected_conv1d(channels=[final_in, int((2*final_in)**0.5), 2], final_proc=False)

    def forward(self, xy, time):
        # -> [bs, 2, n_point]
        xy = xy[:, :2, :]

        # 获取时间步特征
        time_emb = self.time_mlp(time)

        bs, channel, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt
        assert channel == 2

        # -> [bs, channel, n_stroke, stroke_point]
        xy = xy.view(bs, channel, self.n_stk, self.n_stk_pnt)

        # 生成 sparse graph
        # -> [bs, emb, n_stk]
        sparse_graph_up0 = self.point_to_sparse(xy)
        sparse_graph_up0 = self.tm_sp_l0(sparse_graph_up0, time_emb)
        assert sparse_graph_up0.size()[2] == self.n_stk

        # -> [bs, 2, n_point]
        xy = xy.view(bs, channel, n_point)
        assert xy.size()[1] == 2

        # 生成 dense graph
        # -> [bs, emb, n_point]
        dense_graph_up0 = self.point_to_dense(xy)
        dense_graph_up0 = self.tm_dn_l0(dense_graph_up0, time_emb)
        assert dense_graph_up0.size()[2] == n_point

        sparse_graph_up1, dense_graph_up1 = self.sd1(sparse_graph_up0, dense_graph_up0)
        sparse_graph_up1 = self.tm_sp_l1(sparse_graph_up1, time_emb)
        dense_graph_up1 = self.tm_dn_l1(dense_graph_up1, time_emb)

        sparse_graph_up2, dense_graph_up2 = self.sd2(sparse_graph_up1, dense_graph_up1)
        sparse_graph_up2 = self.tm_sp_l2(sparse_graph_up2, time_emb)
        dense_graph_up2 = self.tm_dn_l2(dense_graph_up2, time_emb)

        sparse_global = torch.max(sparse_graph_up2, dim=2)[0]
        dense_global = torch.max(dense_graph_up2, dim=2)[0]

        fea_global = torch.cat([sparse_global, dense_global], dim=1)

        # -> [bs, emb]
        fea_global = self.global_linear(fea_global)

        # decoder
        # 将 sd_graph 融合全局特征
        sparse_fit = fea_global.unsqueeze(2).repeat(1, 1, self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = fea_global.unsqueeze(2).repeat(1, 1, dense_graph_up2.size(2))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        # 一次上采样
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2)
        sparse_graph_down1 = self.tm_sp_dec_l2(sparse_graph_down1, time_emb)  # -> [bs, sp_l2, n_stk]
        dense_graph_down1 = self.tm_dn_dec_l2(dense_graph_down1, time_emb)  # -> [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        # 二次上采样
        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1)
        sparse_graph_down0 = self.tm_sp_dec_l1(sparse_graph_down0, time_emb)
        dense_graph_down0 = self.tm_dn_dec_l1(dense_graph_down0, time_emb)

        sparse_graph_down0 = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph_down0 = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        # 利用之前的信息更新
        sparse_graph, dense_graph = self.sd_up0(sparse_graph_down0, dense_graph_down0)
        sparse_graph = self.tm_sp_dec_l0(sparse_graph, time_emb)
        dense_graph = self.tm_dn_dec_l0(dense_graph, time_emb)

        # 将sparse graph及xy转移到dense graph
        sparse_graph = sparse_graph.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)
        return noise


def test():
    # sketch_data = np.loadtxt('D:\\document\\DeepLearning\\DataSet\\sketch\\sketch_txt\\train\\Bearing\\195f2f0073f6248e5500d871f34c428a_3.txt', delimiter=',')
    #
    # # 2D coordinates
    # coordinates = sketch_data[:, :2]
    #
    # # sketch mass move to (0, 0), x y scale to [-1, 1]
    # coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    # dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    # coordinates = coordinates / dist
    #
    # sketch_data[:, :2] = coordinates
    # sketch_data[-1, 2] = 17
    #
    # # split all strokes
    # strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == 16)[0] + 1)
    #
    # # drop strokes whose points is below min_point
    # filtered_stk = stk_pnt_num_filter(strokes, 8)
    # filtered_stk = stroke_point_dist_filter(filtered_stk, min_dist=1e-3)
    #
    # filtered_stk = np.concatenate(filtered_stk, axis=0)
    # filtered_stk = torch.from_numpy(filtered_stk).float().cuda().unsqueeze(0)
    #
    # plt.clf()
    # for s in strokes:
    #     plt.plot(s[:, 0], -s[:, 1])
    #     plt.scatter(s[:, 0], -s[:, 1])
    # plt.axis('off')
    # plt.show()
    #
    # classifier = SDGraph(16).cuda()
    # cls = classifier(filtered_stk)
    pass


if __name__ == '__main__':
    # test()

    atensor = torch.rand([3, 2, global_defs.n_stk * global_defs.n_stk_pnt]).cuda()
    t1 = torch.randint(0, 1000, (3,)).long().cuda()

    classifier = SDGraph(10).cuda()
    cls11 = classifier(atensor)

    print(cls11.size())

    print('---------------')

