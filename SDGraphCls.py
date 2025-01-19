import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.Dgcnn import DgcnnEncoder
from encoders.utils import full_connected, full_connected_conv2d

import global_defs


def down_sample(dim_in, dim_out, dropout=0.4):
    """
    将 dense graph 的每个笔划的点数调整为原来的 1/2
    只能用于 dgraph 的特征采样
    [bs, emb, n_stk, n_stk_pnt]
    n_stk_pnt 必须能被 2 整除
    :param dim_in:
    :param dim_out:
    :param dropout:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=dim_in,  # 输入通道数 (RGB)
            out_channels=dim_out,  # 输出通道数 (保持通道数不变)
            kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
            stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
            padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
        ),
        nn.BatchNorm2d(dim_out),
        # nn.GELU(),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout2d(dropout)
    )


class DownSample(nn.Module):
    """
    将 dense graph 的每个笔划的点数调整为原来的 1/2
    只能用于 dgraph 的特征采样，每个笔划上的点数必须能被 4 整除
    """
    def __init__(self, dim_in, dim_out, n_stk_pnt, dropout=0.4):
        super().__init__()

        assert n_stk_pnt % 4 == 0

        # 卷积层数
        self.n_layers = n_stk_pnt // 4 + 1

        # 计算各层通道递增量
        emb_inc = (dim_out / dim_in) ** (1 / (self.n_layers - 1))

        # 各模块
        self.conv_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()

        local_in = dim_in

        for i in range(self.n_layers - 2):

            local_out = int(local_in * emb_inc)

            self.conv_layers.append(nn.Conv2d(local_in, local_out, (1, 3)))
            self.batch_normals.append(nn.BatchNorm2d(local_out))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2),)
            self.drop_outs.append(nn.Dropout2d(dropout))

            local_in = local_out

        # 最后一次卷积单独处理
        self.outcv = nn.Conv2d(local_in, dim_out, (1, 3))
        self.outbn = nn.BatchNorm2d(dim_out)
        self.outat = nn.LeakyReLU(negative_slope=0.2),
        self.outdp = nn.Dropout2d(dropout)

    def forward(self, dense_fea):
        """
        :param dense_fea:
        :return: [bs, emb, n_stk, n_stk_pnt]
        """

        for i in range(self.n_layers - 2):
            cv = self.conv_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            dense_fea = dp(at(bn(cv(dense_fea))))

        dense_fea = self.outdp(self.outat(self.outbn(self.outcv(dense_fea))))
        return dense_fea


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    """
    def __init__(self, dense_in, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.dense_to_sparse = nn.Sequential(
            nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
            nn.BatchNorm2d(dense_in),
            nn.LeakyReLU(negative_slope=0.2),
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


class PointToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    """
    def __init__(self, point_dim, sparse_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # 将 DGraph 的数据转移到 SGraph
        mid_dim = int((point_dim * sparse_out) ** 0.5)
        self.point_increase = nn.Sequential(
            nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
            nn.BatchNorm2d(mid_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
            nn.BatchNorm2d(sparse_out),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(dropout),
        )

    def forward(self, xy):
        """
        :param xy: [bs, emb, n_stk * n_stk_pnt]
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

        return xy


class SparseToDense(nn.Module):
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


class SDGraphEncoder(nn.Module):
    def __init__(self, sparse_in, sparse_out, dense_in, dense_out, n_stk, n_stk_pnt):
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # self.dense_to_sparse = DenseToSparse(dense_in, n_stk, n_stk_pnt)
        self.dense_to_sparse = DenseToSparseAttn(sparse_in, dense_in, sparse_in + dense_in, n_stk_pnt)
        self.sparse_to_dense = SparseToDense(n_stk, n_stk_pnt)

        self.sparse_update = DgcnnEncoder(sparse_in + dense_in, sparse_out)
        self.dense_update = DgcnnEncoder(dense_in + sparse_in, int((dense_in * dense_out) ** 0.5))

        # self.sample = DownSample(int((dense_in * dense_out) ** 0.5), dense_out, n_stk_pnt)
        self.sample = down_sample(int((dense_in * dense_out) ** 0.5), dense_out)

    def forward(self, sparse_fea, dense_fea):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_point]
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
        union_dense = union_dense.view(bs, union_dense.size(1), self.n_stk, self.n_stk_pnt)
        union_dense = self.sample(union_dense)
        union_dense = union_dense.view(bs, union_dense.size(1), (self.n_stk * self.n_stk_pnt) // 2)

        return union_sparse, union_dense


class SDGraph(nn.Module):
    def __init__(self, n_class: int):
        """
        :param n_class: 总类别数
        """
        super().__init__()

        print('cls 25.1.15版')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        self.point_to_sparse = PointToSparse(2, sparse_l0, self.n_stk, self.n_stk_pnt)
        self.point_to_dense = DgcnnEncoder(2, dense_l0)

        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2
                                  )

        sparse_glo = sparse_l0 + sparse_l1 + sparse_l2
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / (sparse_glo + dense_glo)) ** (1 / 3)
        outlayer_l0 = sparse_glo + dense_glo
        outlayer_l1 = int(outlayer_l0 * out_inc)
        outlayer_l2 = int(outlayer_l1 * out_inc)
        outlayer_l3 = n_class

        self.linear = full_connected(channels=[outlayer_l0, outlayer_l1, outlayer_l2, outlayer_l3], final_proc=False)

    def forward(self, xy):
        # -> [bs, 2, n_point]
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


def test():
    atensor = torch.rand([3, 2, 30 * 32]).cuda()
    t1 = torch.randint(0, 1000, (3,)).long().cuda()

    classifier = SDGraph(10).cuda()
    cls11 = classifier(atensor)

    print(cls11.size())

    print('---------------')




if __name__ == '__main__':
    test()
    print('---------------')

