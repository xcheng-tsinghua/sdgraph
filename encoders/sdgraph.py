import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils import full_connected, full_connected_conv1d

import global_defs
from encoders.sdgraph_utils import PointToSparse, PointToDense, TimeMerge, SDGraphEncoder, SinusoidalPosEmb


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
            # nn.GELU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(time_emb_dim // 2, time_emb_dim)
        )

        # 将点坐标生成 sdgraph 的初步特征
        self.point_to_sparse = PointToSparse(2, sparse_l0)
        self.point_to_dense = PointToDense(2, dense_l0)
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
        """
        :param xy: [bs, 2, n_skh_pnt]
        :param time: [bs, ]
        :return: [bs, 2, n_skh_pnt]
        """
        xy = xy[:, :2, :]

        # 获取时间步特征
        time_emb = self.time_mlp(time)

        bs, channel, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt
        assert channel == 2

        # 生成 sparse graph
        # -> [bs, emb, n_stk]
        sparse_graph_up0 = self.point_to_sparse(xy)
        sparse_graph_up0 = self.tm_sp_l0(sparse_graph_up0, time_emb)
        assert sparse_graph_up0.size()[2] == self.n_stk

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
    bs = 3
    atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
    t1 = torch.randint(0, 1000, (bs,)).long().cuda()

    classifier = SDGraphSeg().cuda()
    cls11 = classifier(atensor, t1)

    print(cls11.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





