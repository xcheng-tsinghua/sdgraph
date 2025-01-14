import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.Dgcnn import DgcnnEncoder
from encoders.utils import full_connected

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
        nn.GELU(),
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
            self.activates.append(nn.GELU())
            self.drop_outs.append(nn.Dropout2d(dropout))

            local_in = local_out

        # 最后一次卷积单独处理
        self.outcv = nn.Conv2d(local_in, dim_out, (1, 3))
        self.outbn = nn.BatchNorm2d(dim_out)
        self.outat = nn.GELU()
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
    def __init__(self, dense_dim, sparse_dim, dropout=0.4):
        super().__init__()

        # 将 DGraph 的数据转移到 SGraph
        ps_mid = int((dense_dim * sparse_dim) ** 0.5)
        self.dense_to_sparse = nn.Sequential(
            nn.Conv2d(in_channels=dense_dim, out_channels=ps_mid, kernel_size=(1, 3)),
            nn.BatchNorm2d(ps_mid),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=ps_mid, out_channels=sparse_dim, kernel_size=(1, 3)),
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


class SDGraphEncoder(nn.Module):
    def __init__(self, sparse_in, sparse_out, dense_in, dense_out, n_stk, n_stk_pnt):
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.dense_to_sparse = DenseToSparse(dense_in, dense_in)

        self.sparse_update = DgcnnEncoder(sparse_in + dense_in, sparse_out)
        self.dense_update = DgcnnEncoder(dense_in + sparse_in, int((dense_in * dense_out) ** 0.5))

        self.sample = DownSample(int((dense_in * dense_out) ** 0.5), dense_out, n_stk_pnt)

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

        # -> [bs, emb, n_stk, n_stk_pnt]
        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = self.dense_to_sparse(dense_fea)
        assert sparse_feas_from_dense.size()[2] == self.n_stk

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)
        assert union_sparse.size()[2] == self.n_stk

        # -> [bs, emb, n_stk, n_stk_pnt]
        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        assert dense_feas_from_sparse.size()[2] == self.n_stk and dense_feas_from_sparse.size()[3] == self.n_stk_pnt

        # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)
        assert union_dense.size()[2] == self.n_stk and union_dense.size()[3] == self.n_stk_pnt

        # -> [bs, emb, n_stk * n_stk_pnt]
        union_dense = union_dense.view(bs, union_dense.size(1), self.n_stk * self.n_stk_pnt)

        # update sparse fea
        union_sparse = self.sparse_update(union_sparse)

        # update dense fea
        union_dense = self.dense_update(union_dense)

        # down sample
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

        print('cls 25.1.14版')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        # self.point_to_sparse = PointNet2Encoder(0, 32, 2)
        self.point_to_sparse = DenseToSparse(2, 32)
        self.point_to_dense = DgcnnEncoder(2, 32)

        self.sd1 = SDGraphEncoder(32, 512, 32, 512,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt
                                  )

        self.sd2 = SDGraphEncoder(512, 1024, 512, 1024,
                                  n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2
                                  )

        # self.linear = full_connected(channels=[512 * 2, 512, 128, 64, n_class])

        out_inc = (n_class / (1024 * 2)) ** (1 / 3)
        outlayer_l0 = 1024 * 2
        outlayer_l1 = int(outlayer_l0 * out_inc)
        outlayer_l2 = int(outlayer_l1 * out_inc)
        outlayer_l3 = n_class

        self.linear = full_connected(channels=[outlayer_l0, outlayer_l1, outlayer_l2, outlayer_l3], final_proc=False)

    def forward(self, xy):
        # -> [bs, 2, n_point]
        xy = xy[:, :2, :]

        bs, channel, n_point = xy.size()
        assert n_point == self.n_stk * self.n_stk_pnt
        assert channel == 2

        # -> [bs, channel, n_stroke, stroke_point]
        xy = xy.view(bs, channel, self.n_stk, self.n_stk_pnt)

        # 生成 sparse graph
        sparse_graph = self.point_to_sparse(xy)
        assert sparse_graph.size()[2] == self.n_stk

        # -> [bs, 2, n_point]
        xy = xy.view(bs, channel, n_point)
        assert xy.size()[1] == 2

        # -> [bs, emb, n_point]
        dense_graph = self.point_to_dense(xy)
        assert dense_graph.size()[2] == n_point

        sparse_graph, dense_graph = self.sd1(sparse_graph, dense_graph)
        sparse_graph, dense_graph = self.sd2(sparse_graph, dense_graph)

        sparse_fea = torch.max(sparse_graph, dim=2)[0]
        dense_fea = torch.max(dense_graph, dim=2)[0]

        all_fea = torch.cat([sparse_fea, dense_fea], dim=1)

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

