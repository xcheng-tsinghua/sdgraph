"""
STK2 生成测试
标志位需要特殊处理
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_defs
import encoders.sdgraph_utils as su
import encoders.utils as eu


class SparseUpdate(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self, sp_in, sp_out, n_near=2):
        super().__init__()
        self.encoder = su.GCNEncoder(sp_in, sp_out, n_near)

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
        self.encoder = su.GCNEncoder(dn_in, dn_out, n_near)

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
        self.xy_increase = nn.Sequential(
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
            self.time_merge = su.TimeMerge(sparse_out, sparse_out, time_emb_dim)

    def forward(self, xy, time_emb=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 4]， 包含坐标和 one-hot 状态
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = xy.permute(0, 3, 1, 2)

        # xy = xys[:, :2, :, :]
        # s_one_hot = xys[:, 2:, :, :]

        # -> [bs, emb, n_stk, n_stk_pnt]
        xy = self.xy_increase(xy)

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
        self.encoder = su.GCNEncoder(point_dim, emb_dim, n_near)

        self.with_time = with_time
        if self.with_time:
            self.time_merge = su.TimeMerge(emb_dim, emb_dim, time_emb_dim)

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





class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,  # 输入通道数 (RGB)
                out_channels=dim_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt // 2]
        """
        dense_fea = self.conv(dense_fea)
        return dense_fea


class UpSample(nn.Module):
    """
    对sdgraph同时进行上采样
    上采样后笔划数及笔划上的点数均变为原来的2倍
    """
    def __init__(self, dim_in, dim_out, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim_in,  # 输入通道数
                out_channels=dim_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt * 2]
        """
        dense_fea = self.conv(dense_fea)
        return dense_fea


class SparseToDense(nn.Module):
    """
    将sgraph转移到dgraph
    直接拼接到该笔划对应的点
    """
    def __init__(self):
        super().__init__()

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        dense_feas_from_sparse = einops.repeat(sparse_fea, 'b c s -> b c s sp', sp=dense_fea.size(3))

        # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)

        return union_dense


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self):
        super().__init__()

        # self.n_stk = n_stk
        # self.n_stk_pnt = n_stk_pnt

        # self.dense_to_sparse = nn.Sequential(
        #     nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(dense_in),
        #     activate_func(),
        #     nn.Dropout2d(dropout),
        # )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt - 2]
        # dense_fea = self.dense_to_sparse(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = dense_fea.max(3)[0]
        # assert sparse_feas_from_dense.size(2) == self.n_stk

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)

        return union_sparse


class SDGraphEncoder(nn.Module):
    def __init__(self,
                 sparse_in, sparse_out, dense_in, dense_out,  # 输入输出维度
                 sp_near=2, dn_near=10,  # 更新sdgraph的两个GCN中邻近点数目
                 sample_type='down_sample',  # 采样类型
                 with_time=False, time_emb_dim=0,  # 是否附加时间步
                 dropout=0.4
                 ):
        """
        :param sample_type: [down_sample, up_sample, none]
        """
        super().__init__()
        self.with_time = with_time

        self.dense_to_sparse = DenseToSparse()
        self.sparse_to_dense = SparseToDense()

        self.asaas = sparse_in + dense_in
        self.sparse_update = SparseUpdate(sparse_in + dense_in, sparse_out, sp_near)
        self.dense_update = DenseUpdate(dense_in + sparse_in, dense_out, dn_near)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(dense_out, dense_out, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(dense_out, dense_out, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = su.TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)  # 这里dropout不能为零
            self.time_mlp_dn = su.TimeMerge(dense_out, dense_out, time_emb_dim, dropout)  # 这里dropout不能为零

    def forward(self, sparse_fea, dense_fea, time_emb=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk], [bs, emb, n_stk, n_stk_pnt]
        """

        # 信息交换
        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)
        union_dense = self.sparse_to_dense(sparse_fea, dense_fea)

        # 信息更新
        union_sparse = self.sparse_update(union_sparse)
        union_dense = self.dense_update(union_dense)

        # 采样
        union_dense = self.sample(union_dense)

        # 融合时间步特征
        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense


class SDGraphUNet(nn.Module):
    def __init__(self, channel_in=4, channel_out=4, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.0):
        """
        输入是 (x, y, 0, 1) 即绝对坐标和状态的 one-hot, (0, 1) 表示有效，(1, 0) 表示无效
        :param channel_in:
        :param channel_out:
        :param n_stk:
        :param n_stk_pnt:
        :param dropout:
        """
        super().__init__()
        print('diff test2')

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

        '''点坐标 -> 初始 sdgraph 生成层'''
        self.point_to_sparse = PointToSparse(2, sparse_l0, with_time=True, time_emb_dim=time_emb_dim)
        self.point_to_dense = PointToDense(2, dense_l0, with_time=True, time_emb_dim=time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0 * 2, sparse_l1, dense_l0 * 2, dense_l1,
                                       sp_near=2, dn_near=50,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                       sp_near=2, dn_near=50,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        '''全局特征生成层'''
        global_in = sparse_l2 + dense_l2
        self.global_linear = eu.MLP(dimension=0,
                                    channels=(global_in, int((global_in * global_dim) ** 0.5), global_dim),
                                    final_proc=True,
                                    dropout=dropout
                                    )

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        '''最终输出层'''
        final_in = dense_l0 * 2 + sparse_l0 * 2 + dense_l1 + sparse_l1 + 2
        self.xy_linear = eu.MLP(dimension=1,
                                channels=(final_in, 128, 64, 2),
                                final_proc=False,
                                dropout=dropout)

        self.s_linear = eu.MLP(dimension=1,
                               channels=(final_in, 128, 64, 2),
                               final_proc=False,
                               dropout=dropout)

    def img_size(self):
        return self.n_stk, self.n_stk_pnt, self.channel_out

    def forward(self, xys, time):
        """
        :param xys: [bs, n_stk, n_stk_pnt, channel_in], s 可以是 softmax 获得的软编码
        :param time: [bs, ]
        :return: [bs, n_stk, n_stk_pnt, channel_out]
        """
        '''生成时间步特征'''
        time_emb = self.time_encode(time)

        bs, n_stk, n_stk_pnt, channel_in = xys.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_in == self.channel_in

        '''生成初始 sdgraph'''

        xy = xys[..., :2]
        s_one_hot = xys[..., 2:]

        sparse_graph_xy_up0 = self.point_to_sparse(xy, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_xy_up0 = self.point_to_dense(xy, time_emb)  # -> [bs, emb, n_point]

        sparse_graph_s_up0 = self.point_to_sparse(s_one_hot, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_s_up0 = self.point_to_dense(s_one_hot, time_emb)  # -> [bs, emb, n_point]

        sparse_graph_up0 = torch.cat([sparse_graph_xy_up0, sparse_graph_s_up0], dim=1)
        dense_graph_up0 = torch.cat([dense_graph_xy_up0, dense_graph_s_up0], dim=1)

        '''下采样'''
        sparse_graph_up1, dense_graph_up1 = self.sd_down1(sparse_graph_up0, dense_graph_up0, time_emb)
        sparse_graph_up2, dense_graph_up2 = self.sd_down2(sparse_graph_up1, dense_graph_up1, time_emb)

        '''获取全局特征'''
        sp_up2_glo = sparse_graph_up2.max(2)[0]
        dn_up2_glo = dense_graph_up2.amax((2, 3))

        fea_global = torch.cat([sp_up2_glo, dn_up2_glo], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征 (直接拼接在后面)'''
        sparse_fit = einops.repeat(fea_global, 'b c -> b c s', s=self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = einops.repeat(fea_global, 'b c -> b c s sp', s=dense_graph_up2.size(2), sp=dense_graph_up2.size(3))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2, time_emb)  # -> [bs, sp_l2, n_stk], [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1, time_emb)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = einops.repeat(sparse_graph, 'b c s -> b c s sp', sp=self.n_stk_pnt)
        xy = xy.permute(0, 3, 1, 2)  # -> [bs, channel, n_stk, n_stk_pnt]

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.xy_linear(dense_graph)  # -> [bs, channel_out, n_stk * n_stk_pnt]
        noise = einops.rearrange(noise, 'b c (s sp) -> b s sp c', s=self.n_stk, sp=self.n_stk_pnt)

        s_pred = self.s_linear(dense_graph)
        s_pred = F.softmax(s_pred, dim=1)
        s_pred = einops.rearrange(s_pred, 'b c (s sp) -> b s sp c', s=self.n_stk, sp=self.n_stk_pnt)

        s_noise_pred = torch.cat([noise, s_pred], dim=-1)

        return s_noise_pred


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
    atensor = torch.rand([bs, global_defs.n_stk, global_defs.n_stk_pnt, 4])
    t1 = torch.randint(0, 1000, (bs,)).long()

    classifier_unet = SDGraphUNet(4, 4)
    cls11 = classifier_unet(atensor, t1)
    print(cls11.size())

    # classifier_cls = SDGraphCls(10)
    # cls12 = classifier_cls(atensor)
    # print(cls12.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





