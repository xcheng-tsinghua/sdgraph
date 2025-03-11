import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from encoders.utils import full_connected_conv1d, full_connected_conv2d, activate_func, fps, index_points, square_distance
import global_defs


class SDGraphEncoder(nn.Module):
    def __init__(self,
                 sparse_in, sparse_out, dense_in, dense_out,  # 输入输出维度
                 n_stk, n_stk_pnt,  # 笔划数，每个笔划中的点数
                 sp_near=10, dn_near=10,  # 更新sdgraph的两个GCN中邻近点数目
                 sample_type='down_sample',  # 采样类型
                 with_time=False, time_emb_dim=0,  # 是否附加时间步
                 dropout=0.4
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

        self.sparse_update = GCNEncoder(sparse_in + dense_in, sparse_out, sp_near, dropout=0.0)
        self.dense_update = GCNEncoder(dense_in + sparse_in, int((dense_in * dense_out) ** 0.5), dn_near, dropout)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(int((dense_in * dense_out) ** 0.5), dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)
            self.time_mlp_dn = TimeMerge(dense_out, dense_out, time_emb_dim, dropout)

    def forward(self, sparse_fea, dense_fea, time_emb=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_point]
        :param time_emb: [bs, emb]
        :return:
        """
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

        # 下采样
        union_dense = self.sample(union_dense)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense


class SDGraphEncoderUNet(nn.Module):
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

        self.sparse_update = GCNEncoder(sparse_in + dense_in, sparse_out, sp_near, dropout)
        self.dense_update = GCNEncoder(dense_in + sparse_in, dense_out, dn_near, dropout)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample2(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'up_sample':
            self.sample = UpSample2(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)
            self.time_mlp_dn = TimeMerge(dense_out, dense_out, time_emb_dim, dropout)

    def forward(self, sparse_fea, dense_fea, time_emb=None, stk_fea=None, stk_fea_bef=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_point]
        :param stk_fea: 初始用点获取的每个笔划的特征
        :param stk_fea_bef: 采样前的笔划特征
        :param time_emb: [bs, emb]
        :return:
        """
        # 确保在上采样时stk_fea_bef不为None，且下采样时stk_fea_bef为None， ^ :异或，两者不同为真
        assert (self.sample_type == 'up_sample') ^ (stk_fea_bef is None)

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
        union_sparse, union_dense, stk_fea_sampled = self.sample(union_sparse, union_dense, stk_fea, stk_fea_bef, 3)
        if stk_fea_sampled is not None:
            assert stk_fea_sampled.size(0) == bs and (stk_fea_sampled.size(2) == self.n_stk // 2 or stk_fea_sampled.size(2) == self.n_stk * 2)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense, stk_fea_sampled


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

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
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
    def __init__(self, emb_in, emb_out, n_near=10, dropout=0.4):
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
                                           drop_rate=dropout
                                           )
        self.conv2 = full_connected_conv2d([emb_l2_0, emb_l2_1, emb_l2_2],
                                           final_proc=True,
                                           drop_rate=dropout
                                           )

        self.conv3 = full_connected_conv1d([emb_l3_0, emb_l3_1, emb_l3_2],
                                           final_proc=True, drop_rate=dropout
                                           )

    def forward(self, x):
        # fea: [bs, channel, n_token]

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
    def __init__(self, dim_in, dim_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,  # 输入通道数 (RGB)
                out_channels=dim_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_pnt]
        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        dense_fea = self.conv(dense_fea)
        dense_fea = dense_fea.view(bs, dense_fea.size(1), (self.n_stk * self.n_stk_pnt) // 2)

        return dense_fea


class DownSample2(nn.Module):
    """
    对sdgraph同时进行下采样
    该模块处理后笔划数和笔划中的点数同时降低为原来的1/2
    """
    def __init__(self, sp_in, sp_out, dn_in, dn_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.sp_conv = full_connected_conv1d([sp_in, sp_out], True, dropout, True)

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

    def forward(self, sparse_fea, dense_fea, stk_fea, stk_fea_bef=None, n_sp_up_near=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_pnt]
        :param stk_fea: [bs, emb, n_stk] 每个笔划的原始特征，用于进行采样
        :param stk_fea_bef: 占位
        :param n_sp_up_near: 占位
        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        n_stk = sparse_fea.size(2)
        assert n_stk == self.n_stk

        # 使用FPS采样
        stk_fea = stk_fea.permute(0, 2, 1)
        fps_idx = fps(stk_fea, self.n_stk // 2)  # -> [bs, n_stk // 2]

        # 然后找到特征空间中与之最近的笔划进行maxPooling
        knn_idx = knn(stk_fea.permute(0, 2, 1), 2)  # -> [bs, n_stk, 2]
        assert knn_idx.size(2) == 2
        knn_idx = index_points(knn_idx, fps_idx)  # -> [bs, n_stk // 2, 2]
        assert knn_idx.size(1) == self.n_stk // 2
        sparse_fea = index_points(sparse_fea.permute(0, 2, 1), knn_idx).permute(0, 3, 1, 2)  # -> [bs, emb, n_stk // 2, 2]
        assert sparse_fea.size(2) == self.n_stk // 2 and sparse_fea.size(3) == 2
        sparse_fea = sparse_fea.max(3)[0]  # -> [bs, emb, n_stk // 2]
        assert sparse_fea.size(2) == self.n_stk // 2
        sparse_fea = self.sp_conv(sparse_fea)
        stk_fea_sampled = index_points(stk_fea, fps_idx).permute(0, 2, 1)  # -> [bs, emb, n_stk // 2]
        assert stk_fea_sampled.size(2) == self.n_stk // 2

        # 对dense fea进行下采样
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        dense_fea = self.dn_conv(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt // 2]
        dense_fea_emb = dense_fea.size(1)
        dense_fea = dense_fea.permute(0, 2, 3, 1)  # -> [bs, n_stk, n_stk_pnt // 2, emb]
        assert dense_fea.size(2) == self.n_stk_pnt // 2 and dense_fea.size(1) == self.n_stk
        dense_fea = dense_fea.reshape(bs, self.n_stk, (self.n_stk_pnt // 2) * dense_fea_emb)  # -> [bs, n_stk, (n_stk_pnt // 2) * emb]

        # 取对应部分
        dense_fea = index_points(dense_fea, knn_idx)  # -> [bs, n_stk // 2, 2, (n_stk_pnt // 2) * emb]
        assert dense_fea.size(1) == self.n_stk // 2
        dense_fea = dense_fea.view(bs, self.n_stk // 2, 2, self.n_stk_pnt // 2, dense_fea_emb)  # -> [bs, n_stk // 2, 2, n_stk_pnt // 2, emb]
        dense_fea = dense_fea.max(2)[0]  # -> [bs, n_stk // 2, n_stk_pnt // 2, emb]

        dense_fea = dense_fea.view(bs, (self.n_stk * self.n_stk_pnt) // 4, dense_fea_emb).permute(0, 2, 1)  # -> [bs, emb, n_skh_pnt // 4]

        return sparse_fea, dense_fea, stk_fea_sampled


class UpSample2(nn.Module):
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


class UpSample(nn.Module):
    """
    对sdgraph同时进行上采样
    上采样后笔划数及笔划上的点数均变为原来的2倍
    """
    def __init__(self, dim_in, dim_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim_in,  # 输入通道数
                out_channels=dim_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_pnt]
        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        dense_fea = self.conv(dense_fea)
        dense_fea = dense_fea.view(bs, dense_fea.size(1), (self.n_stk * self.n_stk_pnt) * 2)

        return dense_fea


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
        :param dense_fea: [bs, emb, n_stk * n_stk_pnt]
        :return: [bs, emb, n_stk * n_stk_pnt]
        """
        bs, emb, n_pnt = dense_fea.size()
        dense_fea = dense_fea.view(bs, emb, global_defs.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk, 1]
        sparse_fea = sparse_fea.unsqueeze(3)
        assert sparse_fea.size(2) == global_defs.n_stk

        dn_q = self.dn_q(dense_fea)  # -> [bs, emb, n_stk, 1]
        sp_k = self.sp_k(sparse_fea)  # -> [bs, emb, n_stk, n_stk_pnt]
        dn_v = self.dn_v(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt]

        # y_i = emb * F.softmax(self.gamma(fai_xi - psi_xj), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        coef = F.softmax(self.gamma(dn_q - sp_k), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        dn_v = coef * dn_v  # -> [bs, emb, n_stk, n_stk_pnt]

        dn_v = dn_v.view(bs, self.emb_final, n_pnt)

        return dn_v


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
        :param dense_fea: [bs, emb, n_stk * n_stk_pnt]
        :return: [bs, emb, n_stk]
        """
        bs, emb, n_pnt = dense_fea.size()
        dense_fea = dense_fea.view(bs, emb, global_defs.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk, 1]
        sparse_fea = sparse_fea.unsqueeze(3)
        assert sparse_fea.size(2) == global_defs.n_stk

        sp_q = self.sp_q(sparse_fea)  # -> [bs, emb, n_stk, 1]
        dn_k = self.dn_k(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = self.sp_v(sparse_fea)  # -> [bs, emb, n_stk, 1]

        coef = F.softmax(self.gamma(sp_q - dn_k), dim=1)  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = coef * sp_v  # -> [bs, emb, n_stk, n_stk_pnt]
        sp_v = torch.sum(sp_v, dim=3)  # -> [bs, emb, n_stk]

        # union_sparse = torch.cat([sparse_fea.squeeze(), sp_v], dim=1)

        return sp_v


class PointToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph'
    使用一维卷积及最大池化方式
    """
    def __init__(self, point_dim, sparse_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.4,
                 with_time=False, time_emb_dim=0):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        # 将 DGraph 的数据转移到 SGraph
        mid_dim = int((point_dim * sparse_out) ** 0.5)
        self.point_increase = nn.Sequential(
            nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
            nn.BatchNorm2d(mid_dim),
            activate_func(),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
            nn.BatchNorm2d(sparse_out),
            activate_func(),
            nn.Dropout2d(dropout),
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


class PointToDense(nn.Module):
    """
    利用点坐标生成 dense graph
    使用DGCNN直接为每个点生成对应特征
    """
    def __init__(self, point_dim, emb_dim, with_time=False, time_emb_dim=0, n_near=10, dropout=0.):
        super().__init__()
        self.encoder = GCNEncoder(point_dim, emb_dim, n_near, dropout=dropout)

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


# class Block(nn.Module):
#     def __init__(self, dim, dim_out, dropout=0.):
#         super().__init__()
#         self.conv = nn.Conv1d(dim, dim_out, 1)
#         self.norm = nn.BatchNorm1d(dim_out)
#         self.act = activate_func()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, scale_shift=None):
#         """
#         :param x: [bs, channel, n_node]
#         :param scale_shift: [bs, ]
#         :return:
#         """
#         x = self.conv(x)
#         x = self.norm(x)
#
#         if scale_shift is not None:
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift
#
#         x = self.dropout(self.act(x))
#         return x


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


def cross_sample(torch_array, dim, pool='none'):
    """
    将torch_array的dim维度拆分，奇数索引数据和偶数索引数据分别为放在两个数组，之后将这两个数组对应位置取pool
    :param torch_array:
    :param dim:
    :param pool: none: 不池化, max_pool: 最大池化, avg_pool: 平均池化
    :return:
    """
    device = torch_array.device
    target_dim_size = torch_array.size(dim)

    # 确保拆分维度的大小为偶数
    if target_dim_size % 2 != 0:
        raise ValueError(f"指定的维度{dim}的大小必须为偶数，以便进行奇偶拆分。")

    # 先取奇数索引
    odd_data = torch_array.index_select(dim, torch.arange(0, target_dim_size, step=2, device=device))
    even_data = torch_array.index_select(dim, torch.arange(1, target_dim_size, step=2, device=device))

    if pool == 'none':
        res = odd_data
    else:
        odd_data = odd_data.unsqueeze(0)
        even_data = even_data.unsqueeze(0)

        combined_data = torch.cat([odd_data, even_data], dim=0)

        if pool == 'max_pool':
            res = combined_data.max(0)[0]
        elif pool == 'avg_pool':
            res = combined_data.mean(0)
        else:
            raise TypeError('error pool type')

    return res

