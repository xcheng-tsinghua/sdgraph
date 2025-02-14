import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils import full_connected, full_connected_conv2d

import encoders.sdgraph_utils as sdutils
import global_defs


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int):
        """
        :param n_class: 总类别数
        """
        super().__init__()

        print('cls 25.2.14版')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        self.point_to_sparse = sdutils.PointToSparse(2, sparse_l0, self.n_stk, self.n_stk_pnt)
        self.point_to_dense = sdutils.PointToDense(2, dense_l0)

        self.sd1 = sdutils.SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                          n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt
                                          )

        self.sd2 = sdutils.SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
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





