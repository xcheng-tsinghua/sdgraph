# toolkit
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# defined
import encoders.utils as utils


def sample_and_group(xyz, fea, n_center, n_near):
    """
    :param xyz: [bs, n_point, dim]
    :param fea: [bs, n_point, emb]
    :param n_center: fps sample center
    :param n_near: neighbor of each sample point
    :return:
    """
    bs, _, dim = xyz.shape

    # -> [bs, n_center]
    fps_idx = utils.fps_2d(xyz, n_center)

    # -> [bs, n_center, 3]
    new_xyz = utils.index_points(xyz, fps_idx)

    # -> [bs, n_point, n_near]
    knn_idx = utils.knn(xyz, n_near)

    # -> [bs, n_center, n_near]
    knn_idx_of_fps = utils.index_points(knn_idx, fps_idx)

    # -> [bs, n_center, n_near, 3]
    grouped_xyz = utils.index_points(xyz, knn_idx_of_fps)

    # -> dir of center to near
    # -> [bs, n_center, n_near, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(bs, n_center, 1, dim)

    if fea is not None:
        grouped_points = utils.index_points(fea, knn_idx_of_fps)
        new_fea = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_fea = grouped_xyz_norm

    return new_xyz, new_fea


def sample_and_group_all(xyz, fea):
    """
    :param xyz: [bs, n_point, dim]
    :param fea: [bs, n_point, emb]
    :return: [bs, n_point, dim + emb]
    """
    device = xyz.device

    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if fea is not None:
        new_fea = torch.cat([grouped_xyz, fea.view(B, 1, N, -1)], dim=-1)
    else:
        new_fea = grouped_xyz

    return new_xyz, new_fea


class PointNetSetAbstraction(nn.Module):
    """
    set abstraction 层
    包含sampling、grouping、PointNet层
    """
    def __init__(self, in_channel, mlp, group_all, n_near=10):
        """
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        :param group_all: 是否将全部特征集中到一个点
        """
        super().__init__()
        self.group_all = group_all
        self.n_near = n_near

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xy, fea):
        """
        :param xy: [bs, dim, n_point]
        :param fea: [bs, emb, n_point]
        :return:
        """
        assert xy.size()[1] == 2

        n_points = xy.size()[2]
        n_center = n_points // 2 + 1

        xy = xy.permute(0, 2, 1)
        if fea is not None:
            fea = fea.permute(0, 2, 1)

        if self.group_all:
            new_xy, new_fea = sample_and_group_all(xy, fea)
        else:
            new_xy, new_fea = sample_and_group(xy, fea, n_center, self.n_near)

        # -> [bs, dim + emb, n_near, n_center]
        new_fea = new_fea.permute(0, 3, 2, 1).to(torch.float)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_fea = F.relu(bn(conv(new_fea)))

        # -> [bs, emb, n_center]
        new_fea = torch.max(new_fea, 2)[0]

        # -> [bs, 2, n_center]
        new_xy = new_xy.permute(0, 2, 1)

        return new_xy, new_fea


class PointNet2Encoder(nn.Module):
    """
    encode a list of 2D points into a vector
    """
    def __init__(self, emb_in, emb_out, pnt_dim, final_proc=True):
        """
        :param pnt_dim: the channel number of input points
        :param emb_in: the feature number of input points
        :param emb_out: the channel number of output points
        """
        super().__init__()
        in_channel = pnt_dim + emb_in

        emb_inc = (1.2 * emb_out / in_channel) ** (1 / 6)

        emb_l0_0 = in_channel
        emb_l0_1 = int(emb_l0_0 * emb_inc)
        emb_l0_2 = int(emb_l0_1 * emb_inc)
        emb_l0_3 = int(emb_l0_2 * emb_inc)

        emb_l1_0 = emb_l0_3 + pnt_dim
        emb_l1_1 = int(emb_l1_0 * emb_inc)
        emb_l1_2 = int(emb_l1_1 * emb_inc)
        emb_l1_3 = int(1.2 * emb_out)

        self.sa1 = PointNetSetAbstraction(in_channel=emb_l0_0, mlp=[emb_l0_1, emb_l0_2, emb_l0_3],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(in_channel=emb_l1_0, mlp=[emb_l1_1, emb_l1_2, emb_l1_3],
                                          group_all=True)

        self.fc1 = utils.full_connected(channels=[emb_l1_3, int(emb_out / 1.2**0.5), emb_out],
                                        final_proc=final_proc,
                                        drop_rate=0
                                        )


        # emb_inc = (emb_out - in_channel) // 3
        # emb_l1 = in_channel + emb_inc
        # embc_l1 = (emb_l1 + in_channel) // 2
        #
        # emb_l2 = in_channel + 2 * emb_inc
        # embc_l2 = (emb_l2 + emb_l1) // 2
        #
        # self.sa1 = PointNetSetAbstraction(in_channel=in_channel, mlp=[embc_l1, embc_l1+4, emb_l1],
        #                                   group_all=False)
        # self.sa2 = PointNetSetAbstraction(in_channel=emb_l1 + pnt_dim, mlp=[embc_l2, embc_l2+8, emb_l2],
        #                                   group_all=True)
        #
        # self.fc1 = utils.full_connected(channels=[emb_l2, (emb_l2 + emb_out) // 2, emb_out], final_proc=final_proc, drop_rate=0)

    def forward(self, xy, fea=None):
        """
        :param xy: [bs, dim, n_point]
        :param fea: [bs, emb, n_point]
        :return:
        """
        bs, channel, _ = xy.shape
        assert channel == 2

        l1_xyz, l1_fea = self.sa1(xy, fea)
        _, l2_fea = self.sa2(l1_xyz, l1_fea)

        l2_fea = l2_fea.view(bs, -1)

        # -> [bs, emb]
        l2_fea = self.fc1(l2_fea)
        assert len(l2_fea.size()) == 2

        return l2_fea


class PointNet2(nn.Module):
    """
    encode a list of 2D points into a vector
    """
    def __init__(self, n_classes):
        super().__init__()

        self.encoder = PointNet2Encoder(0, 512, 2)
        self.full = utils.full_connected([512, 256, 128, 64, n_classes])

    def forward(self, xy, fea=None):
        fea = self.encoder(xy, fea)
        fea = self.full(fea)

        fea = F.log_softmax(fea, dim=1)

        return fea


if __name__ == '__main__':
    aencod = PointNet2Encoder(10, 2056, 3)

    axy = torch.rand([2, 3, 2048])
    aemb = torch.rand([2, 10, 2048])

    emb = aencod(axy, aemb)

    print(emb.size())

    pass
