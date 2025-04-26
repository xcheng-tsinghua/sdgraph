import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_defs
from encoders import sdgraph_utils as su
from encoders import utils as eu


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
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, sparse_fea, dense_fea, stk_coor, stk_fea_bef=None, n_stk_center=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: [bs, n_stk, emb]
        :param stk_fea_bef: 占位，因为上采样时和在采样时的输入不同
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
        # 先找到对应的笔划
        dense_fea = einops.rearrange(dense_fea, 'b c s sp -> b s (sp c)')
        dense_fea = eu.index_points(dense_fea, fps_idx)  # -> [bs, n_stk // 2, n_stk_pnt * emb]
        dense_fea = einops.rearrange(dense_fea, 'b s (sp c) -> b c s sp', sp=self.n_stk_pnt)

        # 进行下采样
        dense_fea = self.dn_conv(dense_fea)  # -> [bs, emb, n_stk // 2, n_stk_pnt // 2]
        return sparse_fea, dense_fea, stk_coor_sampled


class UpSample(nn.Module):
    def __init__(self, sp_in, sp_out, dn_in, dn_out, n_stk, n_stk_pnt, sp_near=2, dropout=0.4):
        """

        :param sp_in:
        :param sp_out:
        :param dn_in:
        :param dn_out:
        :param n_stk:
        :param n_stk_pnt:
        :param sp_near: sparse graph 上采样过程中搜寻的附近点点数
        :param dropout:
        """
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.sp_near = sp_near

        self.sp_conv = eu.MLP(dimension=1, channels=(sp_in, sp_out), dropout=dropout, final_proc=True)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dn_in,  # 输入通道数
                out_channels=dn_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dn_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, sparse_fea, dense_fea, stk_coor, stk_coor_bef, n_stk_center=None):
        """
        对于 stk_coor_bef 中的某个点(center)，找到 stk_coor 中与之最近的 sp_near 个点(nears)，将nears的特征进行加权和，得到center的插值特征
        nears中第i个点(near_i)特征的权重为 [1/d(near_i)]/sum(k=1->n_sp_up_near)[1/d(near_k)]
        d(near_i)为 center到near_i的距离，即距离越近，权重越大
        之后拼接 sparse_fea_bef 与插值后的 sparse_fea，再利用 MLP 对每个点的特征单独进行处理

        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: 采样后的笔划特征 [bs, n_stk, emb]
        :param stk_coor_bef: 采样前的笔划特征 [bs, n_stk * 2, emb]
        :param n_stk_center: 占位用
        :return:
        """
        bs, _, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt

        # ----- 对sgraph进行上采样 -----
        # 计算sparse_fea_bef中的每个点到sparse_fea中每个点的平方距离
        # sparse_fea_bef:[bs, n_stk_bef, emb]
        # sparse_fea:[bs, n_stk, emb]
        # dists -> [bs, n_stk_bef, n_stk]
        dists = eu.square_distance(stk_coor_bef, stk_coor)
        assert dists.size(1) == stk_coor_bef.size(1) and dists.size(2) == self.n_stk

        # 计算每个初始点到采样点距离最近的3个点，sort 默认升序排列, 取三个
        dists, idx = dists.sort(dim=2)
        dists, idx = dists[:, :, :self.sp_near], idx[:, :, :self.sp_near]  # [bs, n_stk_bef, sp_near]

        # 最近距离的每行求倒数，距离越近，数值越大
        dist_recip = 1.0 / (dists + 1e-8)

        # 求倒数后，每个中心点的附近点平方距离，除以属于该中心点的所有附近点平方距离之和
        norm = torch.sum(dist_recip, dim=2, keepdim=True)  # ->[bs, n_stk_bef, 1]
        weight = dist_recip / norm  # ->[bs, n_stk * 2, sp_near]

        # 找到中心点附近的 sp_near 个最近的邻近点特征
        sparse_fea = eu.index_points(sparse_fea, idx, True)  # -> [bs, emb, n_stk_bef, sp_near]

        # 将邻近点特征进行加权求和，之后使用 MLP 更新特征
        sparse_fea = (sparse_fea * weight.unsqueeze(1)).sum(3)  # -> [bs, emb, n_stk_bef]
        assert sparse_fea.size(2) == stk_coor_bef.size(1)
        sparse_fea = self.sp_conv(sparse_fea)

        # ----- 为 dense graph 进行笔划层级上采样 -----
        dense_fea = einops.rearrange(dense_fea, 'b c s sp -> b s (sp c)')
        dense_fea = eu.index_points(dense_fea, idx)  # -> [bs, n_stk_bef, sp_near, n_stk_pnt * emb]
        dense_fea = torch.sum(dense_fea * weight.unsqueeze(3), dim=2)  # -> [bs, n_stk_bef, n_stk_pnt * emb]
        dense_fea = einops.rearrange(dense_fea, 'b s (sp c) -> b c s sp', sp=n_stk_pnt)
        dense_fea = self.conv(dense_fea)  # -> [bs, emb, n_stk_bef, n_stk_pnt * 2]

        return sparse_fea, dense_fea, None


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

        self.encoder = eu.MLP(
            dimension=1,
            channels=(sparse_in + dense_in, sparse_in),
            dropout=dropout,
            final_proc=True
        )

        # self.n_stk = n_stk
        # self.n_stk_pnt = n_stk_pnt

        # self.dense_to_sparse = nn.Sequential(
        #     nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(dense_in),
        #     eu.activate_func(),
        #     nn.Dropout2d(dropout),
        # )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        # bs, emb, _ = dense_fea.size()
        #
        # # -> [bs, emb, n_stk, n_stk_pnt]
        # dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk, n_stk_pnt]
        # sparse_feas_from_dense = self.dense_to_sparse(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = dense_fea.max(3)[0]
        # assert sparse_feas_from_dense.size(2) == self.n_stk

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
                 sample_type='down_sample',
                 with_time=False, time_emb_dim=0,
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
        :param sample_type: [down_sample, up_sample, none]
        :param with_time: 是否附加时间步
        :param time_emb_dim: 时间步特征维度
        :param dropout:
        """
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.n_stk_center = n_stk_center
        self.with_time = with_time

        self.dense_to_sparse = DenseToSparse(sparse_in, dense_in, dropout)  # 此处dropout之前测试不能设为零
        self.sparse_to_dense = SparseToDense(sparse_in, dense_in, dropout)  # 此处dropout之前测试不能设为零

        self.sparse_update = su.SparseUpdate(sparse_in, sparse_out, sp_near)
        self.dense_update = su.DenseUpdate(dense_in, dense_out, dn_near, dropout)  # 此处dropout效果未测试

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout=dropout)
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(sparse_out, sparse_out, dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout=dropout)
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = su.TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)
            self.time_mlp_dn = su.TimeMerge(dense_out, dense_out, time_emb_dim, dropout)

    def forward(self, sparse_fea, dense_fea, time_emb=None, stk_coor=None, stk_coor_bef=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param stk_coor: 笔划坐标 [bs, n_stk, emb]
        :param stk_coor_bef: 采样前的笔划坐标 [bs, n_stk // 2, emb]
        :param time_emb: [bs, emb]
        :return:
        """
        # 确保在上采样时stk_coor_bef不为None，且下采样时stk_coor_bef为None， ^ :异或，两者不同为真
        assert (self.sample_type == 'up_sample') ^ (stk_coor_bef is None)

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
        union_sparse, union_dense, stk_coor_sampled = self.sample(union_sparse, union_dense, stk_coor, stk_coor_bef, self.n_stk_center)
        if stk_coor_sampled is not None:
            assert stk_coor_sampled.size(0) == bs and (stk_coor_sampled.size(1) == self.n_stk_center or stk_coor_sampled.size(1) == self.n_stk * 2)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense, stk_coor_sampled


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('sdgraph cls with stk sample')

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # 各层特征维度
        sparse_l0 = 32 + 16
        sparse_l1 = 128 + 64
        sparse_l2 = 512 + 256

        dense_l0 = 32
        dense_l1 = 128
        dense_l2 = 512

        # 生成笔划坐标
        self.point_to_stk_coor = su.PointToSparse(2, sparse_l0)

        # 生成初始 sdgraph
        self.point_to_sparse = su.PointToSparse(2, sparse_l0)
        self.point_to_dense = su.PointToDense(2, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  self.n_stk, self.n_stk_pnt,
                                  self.n_stk // 2, self.n_stk_pnt // 2,
                                  dropout=dropout
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  self.n_stk // 2, self.n_stk_pnt // 2,
                                  self.n_stk // 4, self.n_stk_pnt // 4,
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
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel == 2

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


class SDGraphUNet(nn.Module):
    def __init__(self, channel_in, channel_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.0):
        super().__init__()
        print('sdgraph unet with stk sample')

        '''草图参数'''
        self.channel_in = channel_in
        self.channel_out = channel_out
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
        self.time_encode = su.TimeEncode(time_emb_dim)

        '''生成笔划坐标'''
        self.point_to_stk_coor = su.PointToSparse(channel_in, sparse_l0)

        '''点坐标 -> 初始 sdgraph 生成层'''
        self.point_to_sparse = su.PointToSparse(channel_in, sparse_l0, with_time=True, time_emb_dim=time_emb_dim)
        self.point_to_dense = su.PointToDense(channel_in, dense_l0, with_time=True, time_emb_dim=time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                       self.n_stk, self.n_stk_pnt,
                                       self.n_stk - 1, self.n_stk_pnt // 2,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                       self.n_stk - 1, self.n_stk_pnt // 2,
                                       self.n_stk - 2, self.n_stk_pnt // 4,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        '''全局特征生成层'''
        global_in = sparse_l2 + dense_l2
        self.global_linear = eu.MLP(dimension=0,
                                    channels=(global_in, int((global_in * global_dim) ** 0.5), global_dim),
                                    final_proc=True,
                                    dropout=dropout)

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     self.n_stk - 2, self.n_stk_pnt // 4,
                                     self.n_stk - 1, self.n_stk_pnt // 2,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     self.n_stk - 1, self.n_stk_pnt // 2,
                                     self.n_stk, self.n_stk_pnt,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        '''最终输出层'''
        final_in = dense_l0 + sparse_l0 + dense_l1 + sparse_l1 + channel_in
        self.final_linear = eu.MLP(dimension=1,
                                   channels=(final_in, int((channel_out * final_in) ** 0.5), channel_out),
                                   final_proc=False,
                                   dropout=dropout)

    def img_size(self):
        return self.n_stk, self.n_stk_pnt, self.channel_out

    def forward(self, xy, time):
        """
        :param xy: [bs, channel_in, n_skh_pnt]
        :param time: [bs, ]
        :return: [bs, channel_out, n_skh_pnt]
        """
        '''生成时间步特征'''
        time_emb = self.time_encode(time)

        bs, n_stk, n_stk_pnt, channel_in = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_in == 2

        # 获取笔划坐标，即初始节点坐标
        stk_coor_l0 = self.point_to_stk_coor(xy).permute(0, 2, 1)  # [bs, n_stk, emb]

        '''生成初始 sdgraph'''
        sparse_graph_up0 = self.point_to_sparse(xy, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_up0 = self.point_to_dense(xy, time_emb)  # -> [bs, emb, n_point]
        assert sparse_graph_up0.size()[2] == self.n_stk and dense_graph_up0.size()[2] == n_stk and dense_graph_up0.size()[3] == n_stk_pnt

        '''下采样'''
        sparse_graph_up1, dense_graph_up1, stk_coor_l1 = self.sd_down1(sparse_graph_up0, dense_graph_up0, time_emb, stk_coor_l0)
        sparse_graph_up2, dense_graph_up2, stk_coor_l2 = self.sd_down2(sparse_graph_up1, dense_graph_up1, time_emb, stk_coor_l1)

        '''获取全局特征'''
        sp_up2_glo = sparse_graph_up2.max(2)[0]
        dn_up2_glo = dense_graph_up2.amax((2, 3))

        fea_global = torch.cat([sp_up2_glo, dn_up2_glo], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征 (直接拼接在后面)'''
        sparse_fit = einops.repeat(fea_global, 'b c -> b c s', s=sparse_graph_up2.size(2))
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = einops.repeat(fea_global, 'b c -> b c s sp', s=dense_graph_up2.size(2), sp=dense_graph_up2.size(3))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1, _ = self.sd_up2(sparse_graph_down2, dense_graph_down2, time_emb, stk_coor_l2, stk_coor_l1)  # -> [bs, sp_l2, n_stk], [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0, _ = self.sd_up1(sparse_graph_down1, dense_graph_down1, time_emb, stk_coor_l1, stk_coor_l0)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = einops.repeat(sparse_graph, 'b c n -> b c n sp', sp=self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel_in, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        final_out = self.final_linear(dense_graph)
        final_out = einops.rearrange(final_out, 'b c (s sp) -> b s sp c', s=self.n_stk, sp=self.n_stk_pnt)

        return final_out


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
    t1 = torch.randint(0, 1000, (bs,)).long()

    classifier_unet = SDGraphUNet(2, 2)
    cls11 = classifier_unet(atensor, t1)
    print(cls11.size())

    classifier_cls = SDGraphCls(10)
    cls12 = classifier_cls(atensor)
    print(cls12.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





