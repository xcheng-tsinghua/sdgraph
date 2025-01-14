
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.Dgcnn import DgcnnEncoder
from encoders.PointNet2 import PointNet2Encoder
from encoders.utils import full_connected


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
    """
    AttentionEncoder: update sparse graph
    PointNet2Encoder: translate dense graph feature into sparse graph
    DgcnnEncoder: update dense graph
    """
    def __init__(self, sparse_in, sparse_out, dense_in, dense_out, n_stroke, stroke_point):
        super().__init__()
        self.n_stroke = n_stroke
        self.stroke_point = stroke_point

        # self.dense_to_sparse = PointNet2Encoder(dense_in, dense_in, 2)
        self.dense_to_sparse = DenseToSparse(dense_in, dense_in)

        self.sparse_update = DgcnnEncoder(sparse_in + dense_in, sparse_out)
        self.dense_update = DgcnnEncoder(dense_in + sparse_in, dense_out)

    def forward(self, sparse_fea, dense_fea):
        """
        :param xyz: [bs, 2, n_points]
        :param sparse_fea: [bs, emb, n_stroke]
        :param dense_fea: [bs, emb, n_point]
        :return:
        """
        bs, emb, n_stroke = sparse_fea.size()
        assert n_stroke == self.n_stroke

        n_points = dense_fea.size()[2]
        assert n_points == self.n_stroke * self.stroke_point

        # -> [bs, emb, n_stroke, stroke_point]
        dense_fea = dense_fea.view(bs, emb, self.n_stroke, self.stroke_point)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = self.dense_to_sparse(dense_fea)
        assert sparse_feas_from_dense.size()[2] == self.n_stroke

        # -> [bs, emb, n_stroke]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)
        assert union_sparse.size()[2] == self.n_stroke

        # union sparse fea to dense fea

        # -> [bs, emb, n_stroke, stroke_point]
        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, self.stroke_point)
        assert dense_feas_from_sparse.size()[2] == self.n_stroke and dense_feas_from_sparse.size()[3] == self.stroke_point

        # -> [bs, emb, n_stroke, stroke_point]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)
        assert union_dense.size()[2] == self.n_stroke and union_dense.size()[3] == self.stroke_point

        # -> [bs, emb, n_stroke * stroke_point]
        union_dense = union_dense.view(bs, -1, self.n_stroke * self.stroke_point)

        # update sparse fea
        union_sparse = self.sparse_update(union_sparse)

        # update dense fea
        union_dense = self.dense_update(union_dense)

        return union_sparse, union_dense


class SDGraph(nn.Module):
    def __init__(self, n_class: int, n_stroke: int = 30, stroke_point: int = 32):
        """
        :param n_class: 总类别数
        :param n_stroke: 每个草图中的笔划数
        :param stroke_point: 每个笔划中的点数
        """
        super().__init__()

        print('cls 初始版')

        self.n_stroke = n_stroke
        self.stroke_point = stroke_point

        # self.point_to_sparse = PointNet2Encoder(0, 32, 2)
        self.point_to_sparse = DenseToSparse(2, 32)
        self.point_to_dense = DgcnnEncoder(2, 32)

        self.sd1 = SDGraphEncoder(32, 512, 32, 512,
                                  n_stroke=self.n_stroke, stroke_point=self.stroke_point
                                  )

        self.sd2 = SDGraphEncoder(512, 1024, 512, 1024,
                                  n_stroke=self.n_stroke, stroke_point=self.stroke_point
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
        assert n_point == self.n_stroke * self.stroke_point
        assert channel == 2

        # -> [bs, channel, n_stroke, stroke_point]
        xy = xy.view(bs, channel, self.n_stroke, self.stroke_point)

        # 生成 sparse graph
        # sparse_graph = []
        #
        # for i in range(self.n_stroke):
        #     # -> [bs, emb]
        #     stroke_fea = self.point_to_sparse(xy[:, :, i, :])
        #
        #     # -> [bs, emb, 1]
        #     stroke_fea = stroke_fea.unsqueeze(2)
        #
        #     sparse_graph.append(stroke_fea)
        #
        # # -> [bs, emb, n_stroke]
        # sparse_graph = torch.cat(sparse_graph, dim=2)

        sparse_graph = self.point_to_sparse(xy)
        assert sparse_graph.size()[2] == self.n_stroke

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

