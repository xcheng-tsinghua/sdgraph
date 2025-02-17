import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils import full_connected, full_connected_conv1d

import global_defs
from encoders.sdgraph_utils import PointToSparse, PointToDense, TimeMerge, SDGraphEncoder, TimeEncode


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('cls 25.2.15版-from valid')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        # 各层特征维度
        sparse_l0 = 32 + 16
        sparse_l1 = 128 + 64
        sparse_l2 = 512 + 256

        dense_l0 = 32
        dense_l1 = 128
        dense_l2 = 512

        # 生成初始 sdgraph
        self.point_to_sparse = PointToSparse(2, sparse_l0)
        self.point_to_dense = PointToDense(2, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2
                                  )

        # 利用输出特征进行分类
        sparse_glo = sparse_l0 + sparse_l1 + sparse_l2
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / (sparse_glo + dense_glo)) ** (1 / 3)

        outlayer_l0 = sparse_glo + dense_glo
        outlayer_l1 = int(outlayer_l0 * out_inc)
        outlayer_l2 = int(outlayer_l1 * out_inc)
        outlayer_l3 = n_class

        self.linear = full_connected(channels=[outlayer_l0, outlayer_l1, outlayer_l2, outlayer_l3], final_proc=False)

    def forward(self, xy):
        """
        :param xy: [bs, 2, n_skh_pnt]
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

        # 交叉更新数据
        sparse_graph1, dense_graph1 = self.sd1(sparse_graph0, dense_graph0)
        sparse_graph2, dense_graph2 = self.sd2(sparse_graph1, dense_graph1)

        # 提取全局特征
        sparse_glo0 = torch.max(sparse_graph0, dim=2)[0]
        sparse_glo1 = torch.max(sparse_graph1, dim=2)[0]
        sparse_glo2 = torch.max(sparse_graph2, dim=2)[0]

        dense_glo0 = torch.max(dense_graph0, dim=2)[0]
        dense_glo1 = torch.max(dense_graph1, dim=2)[0]
        dense_glo2 = torch.max(dense_graph2, dim=2)[0]

        all_fea = torch.cat([sparse_glo0, sparse_glo1, sparse_glo2, dense_glo0, dense_glo1, dense_glo2], dim=1)

        # 利用全局特征分类
        cls = self.linear(all_fea)
        cls = F.log_softmax(cls, dim=1)

        return cls


class SDGraphSeg(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        '''草图参数'''
        self.channel_in = channel_in
        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

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
        self.point_to_sparse = PointToSparse(channel_in, sparse_l0)
        self.point_to_dense = PointToDense(channel_in, dense_l0)
        self.tm_sp_l0 = TimeMerge(sparse_l0, sparse_l0, time_emb_dim)
        self.tm_dn_l0 = TimeMerge(dense_l0, dense_l0, time_emb_dim)

        '''下采样层 × 2'''
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt,
                                  sp_near=2, dn_near=10, sample_type='down_sample')
        self.tm_sp_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2,
                                  sp_near=2, dn_near=10, sample_type='down_sample')
        self.tm_sp_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        '''全局特征生成层'''
        self.global_linear = full_connected(
            channels=[sparse_l2 + dense_l2, int(((sparse_l2 + dense_l2) * global_dim) ** 0.5), global_dim],
            final_proc=True)

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 4,
                                     sp_near=2, dn_near=10, sample_type='up_sample')
        self.tm_sp_dec_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_dec_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2,
                                     sp_near=2, dn_near=10, sample_type='up_sample')
        self.tm_sp_dec_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_dec_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        '''混合特征及最终输出层'''
        self.sd_merge = SDGraphEncoder(sparse_l1 + sparse_l0, sparse_l0,
                                     dense_l1 + dense_l0, dense_l0,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt, sample_type='none')
        self.tm_sp_merge = TimeMerge(sparse_l0, sparse_l0, time_emb_dim)
        self.tm_dn_merge = TimeMerge(dense_l0, dense_l0, time_emb_dim)

        final_in = dense_l0 + sparse_l0 + channel_in
        self.final_linear = full_connected_conv1d(channels=[final_in, int((channel_out*final_in)**0.5), channel_out], final_proc=False)

    def pnt_channel(self):
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
        sparse_graph_up0 = self.point_to_sparse(xy)
        sparse_graph_up0 = self.tm_sp_l0(sparse_graph_up0, time_emb)  # -> [bs, emb, n_stk]
        assert sparse_graph_up0.size()[2] == self.n_stk

        dense_graph_up0 = self.point_to_dense(xy)
        dense_graph_up0 = self.tm_dn_l0(dense_graph_up0, time_emb)  # -> [bs, emb, n_point]
        assert dense_graph_up0.size()[2] == n_point

        '''下采样'''
        sparse_graph_up1, dense_graph_up1 = self.sd1(sparse_graph_up0, dense_graph_up0)
        sparse_graph_up1 = self.tm_sp_l1(sparse_graph_up1, time_emb)
        dense_graph_up1 = self.tm_dn_l1(dense_graph_up1, time_emb)

        sparse_graph_up2, dense_graph_up2 = self.sd2(sparse_graph_up1, dense_graph_up1)
        sparse_graph_up2 = self.tm_sp_l2(sparse_graph_up2, time_emb)
        dense_graph_up2 = self.tm_dn_l2(dense_graph_up2, time_emb)

        '''获取全局特征'''
        sparse_global = torch.max(sparse_graph_up2, dim=2)[0]
        dense_global = torch.max(dense_graph_up2, dim=2)[0]

        fea_global = torch.cat([sparse_global, dense_global], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征'''
        sparse_fit = fea_global.unsqueeze(2).repeat(1, 1, self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = fea_global.unsqueeze(2).repeat(1, 1, dense_graph_up2.size(2))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2)
        sparse_graph_down1 = self.tm_sp_dec_l2(sparse_graph_down1, time_emb)  # -> [bs, sp_l2, n_stk]
        dense_graph_down1 = self.tm_dn_dec_l2(dense_graph_down1, time_emb)  # -> [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1)
        sparse_graph_down0 = self.tm_sp_dec_l1(sparse_graph_down0, time_emb)
        dense_graph_down0 = self.tm_dn_dec_l1(dense_graph_down0, time_emb)

        sparse_graph_down0 = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph_down0 = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''融合上采样后的 sdgraph'''
        sparse_graph, dense_graph = self.sd_merge(sparse_graph_down0, dense_graph_down0)
        sparse_graph = self.tm_sp_merge(sparse_graph, time_emb)
        dense_graph = self.tm_dn_merge(dense_graph, time_emb)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = sparse_graph.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel_in, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)
        return noise


class SDGraphSeg2(nn.Module):
    """
    不最终用sdgraphEncoder融合
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        print('2025.2.17 版本')

        '''Diff 参数'''
        self.channels = channel_in
        self.n_pnts = global_defs.n_stk_pnt

        '''草图参数'''
        self.channel_in = channel_in
        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

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
        self.point_to_sparse = PointToSparse(channel_in, sparse_l0)
        self.point_to_dense = PointToDense(channel_in, dense_l0)
        self.tm_sp_l0 = TimeMerge(sparse_l0, sparse_l0, time_emb_dim)
        self.tm_dn_l0 = TimeMerge(dense_l0, dense_l0, time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1, self.n_stk, self.n_stk_pnt,
                                       sp_near=2, dn_near=10, sample_type='down_sample')
        self.tm_sp_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2, self.n_stk, self.n_stk_pnt // 2,
                                       sp_near=2, dn_near=10, sample_type='down_sample')
        self.tm_sp_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        '''全局特征生成层'''
        self.global_linear = full_connected(
            channels=[sparse_l2 + dense_l2, int(((sparse_l2 + dense_l2) * global_dim) ** 0.5), global_dim],
            final_proc=True)

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 4,
                                     sp_near=2, dn_near=10, sample_type='up_sample')
        self.tm_sp_dec_l2 = TimeMerge(sparse_l2, sparse_l2, time_emb_dim)
        self.tm_dn_dec_l2 = TimeMerge(dense_l2, dense_l2, time_emb_dim)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2,
                                     sp_near=2, dn_near=10, sample_type='up_sample')
        self.tm_sp_dec_l1 = TimeMerge(sparse_l1, sparse_l1, time_emb_dim)
        self.tm_dn_dec_l1 = TimeMerge(dense_l1, dense_l1, time_emb_dim)

        '''最终输出层'''
        final_in = dense_l0 + sparse_l0 + dense_l1 + sparse_l1 + channel_in
        self.final_linear = full_connected_conv1d(
            channels=[final_in, int((channel_out * final_in) ** 0.5), channel_out], final_proc=False)

    def pnt_channel(self):
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
        sparse_graph_up0 = self.point_to_sparse(xy)
        sparse_graph_up0 = self.tm_sp_l0(sparse_graph_up0, time_emb)  # -> [bs, emb, n_stk]
        assert sparse_graph_up0.size()[2] == self.n_stk

        dense_graph_up0 = self.point_to_dense(xy)
        dense_graph_up0 = self.tm_dn_l0(dense_graph_up0, time_emb)  # -> [bs, emb, n_point]
        assert dense_graph_up0.size()[2] == n_point

        '''下采样'''
        sparse_graph_up1, dense_graph_up1 = self.sd_down1(sparse_graph_up0, dense_graph_up0)
        sparse_graph_up1 = self.tm_sp_l1(sparse_graph_up1, time_emb)
        dense_graph_up1 = self.tm_dn_l1(dense_graph_up1, time_emb)

        sparse_graph_up2, dense_graph_up2 = self.sd_down2(sparse_graph_up1, dense_graph_up1)
        sparse_graph_up2 = self.tm_sp_l2(sparse_graph_up2, time_emb)
        dense_graph_up2 = self.tm_dn_l2(dense_graph_up2, time_emb)

        '''获取全局特征'''
        sparse_global = torch.max(sparse_graph_up2, dim=2)[0]
        dense_global = torch.max(dense_graph_up2, dim=2)[0]

        fea_global = torch.cat([sparse_global, dense_global], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征'''
        sparse_fit = fea_global.unsqueeze(2).repeat(1, 1, self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = fea_global.unsqueeze(2).repeat(1, 1, dense_graph_up2.size(2))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2)
        sparse_graph_down1 = self.tm_sp_dec_l2(sparse_graph_down1, time_emb)  # -> [bs, sp_l2, n_stk]
        dense_graph_down1 = self.tm_dn_dec_l2(dense_graph_down1, time_emb)  # -> [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1)
        sparse_graph_down0 = self.tm_sp_dec_l1(sparse_graph_down0, time_emb)
        dense_graph_down0 = self.tm_dn_dec_l1(dense_graph_down0, time_emb)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)  # -> [bs, sp_l0 + sp_l1, n_stk]
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)  # -> [bs, dn_l0 + dn_l1, n_pnt]

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = sparse_graph.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel_in, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)
        return noise


def test():
    bs = 3
    atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
    t1 = torch.randint(0, 1000, (bs,)).long().cuda()

    classifier = SDGraphSeg2(2, 2).cuda()
    cls11 = classifier(atensor, t1)

    print(cls11.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





