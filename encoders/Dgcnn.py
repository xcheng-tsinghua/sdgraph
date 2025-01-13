#!/usr/bin/env python
# -*- coding: utf-8 -*-

# packge
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# defined
import encoders.utils as utils


# def get_graph_feature(x, k=20):
#     """
#     输入点云，利用knn计算每个点的邻近点，然后计算内个点中心点到邻近点的向量，再将该向量与中心点坐标拼接
#     :param x: [bs, channel, npoint]
#     :param k:
#     :return: [bs, channel+channel, npoint, k]
#     """
#     bs, channel, npoints = x.size()
#
#     # -> [bs, npoint, 3]
#     x = x.permute(0, 2, 1)
#
#     # -> [bs, npoint, k]
#     idx = utils.knn(x, k)
#
#     # -> [bs, npoint, k, 3]
#     point_neighbors = utils.index_points(x, idx)
#
#     # -> [bs, npoint, k, 3]
#     x = x.view(bs, npoints, 1, channel).repeat(1, 1, k, 1)
#
#     # 计算从中心点到邻近点的向量，再与中心点拼接起来
#     # -> [bs, npoint, k, 3]
#     feature = torch.cat((point_neighbors - x, x), dim=3)
#
#     # -> [bs, 3, npoint, k]
#     feature = feature.permute(0, 3, 1, 2)
#     return feature


def knn(x, k):
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


class DGCNN(nn.Module):
    def __init__(self, output_channels, n_near=10, emb_dims=256, dropout=0.5):
        super(DGCNN, self).__init__()
        # self.k = n_near
        #
        # self.conv1 = utils.full_connected_conv2d([4, 8, 16], final_proc=True, drop_rate=0)
        # self.conv2 = utils.full_connected_conv2d([16*2, 64, 128], final_proc=True, drop_rate=0)
        #
        # self.conv3 = utils.full_connected_conv1d([128 + 16, (128 + 16 + emb_dims) // 2, emb_dims], final_proc=True, drop_rate=0)

        self.encoder = DgcnnEncoder(2, emb_dims)

        self.linear = utils.full_connected([emb_dims, (emb_dims + output_channels) // 2, output_channels], final_proc=False, drop_rate=0)

    def forward(self, x):
        # -> x: [bs, 2, n_point]
        # assert x.size(1) == 2
        #
        # # -> [bs, emb, n_point, n_neighbor]
        # x = get_graph_feature(x, k=self.k)
        # x = self.conv1(x)
        #
        # # -> [bs, emb, n_point]
        # x1 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv2(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        #
        # # -> [bs, emb, n_point]
        # x = torch.cat((x1, x2), dim=1)
        #
        # # -> [bs, emb, n_points]
        # x = self.conv3(x)

        x = self.encoder(x)

        # -> [bs, emb]
        x = torch.max(x, dim=2)[0]

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)
        #
        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        #
        # x = F.log_softmax(x, dim=1)
        # return x


class DgcnnEncoder(nn.Module):
    def __init__(self, emb_in, emb_out, n_near=10, dropout=0):
        super().__init__()
        self.n_near = n_near

        emb_inc = (emb_out / (4*emb_in)) ** 0.25
        emb_l1_0 = emb_in * 2
        emb_l1_1 = int(emb_l1_0 * emb_inc)
        emb_l1_2 = int(emb_l1_0 * emb_inc ** 2)

        emb_l2_0 = emb_l1_2 * 2
        emb_l2_1 = int(emb_l2_0 * emb_inc)
        emb_l2_2 = emb_out

        emb_l3_0 = emb_l2_2 + emb_l1_2
        emb_l3_1 = int(((emb_out / emb_l3_0) ** 0.5) * emb_l3_0)
        emb_l3_2 = emb_out

        self.conv1 = utils.full_connected_conv2d([emb_l1_0, emb_l1_1, emb_l1_2],
                                                 final_proc=True,
                                                 drop_rate=dropout
                                                 )
        self.conv2 = utils.full_connected_conv2d([emb_l2_0, emb_l2_1, emb_l2_2],
                                                 final_proc=True,
                                                 drop_rate=dropout
                                                 )

        self.conv3 = utils.full_connected_conv1d([emb_l3_0, emb_l3_1, emb_l3_2],
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


def test():


    # atensor = torch.ones((2, 2, 2))
    # btensor = torch.rand((2, 1, 1))
    #
    # print(torch.arange(0, 5))
    # print(btensor)
    # print(atensor + btensor)
    # exit()

    # btensor = torch.rand((4, 3))
    # print(btensor)
    # print(btensor.max(dim=-1)[0])
    # exit()

    btensor = torch.rand((2, 256, 100))
    modelaaa = DgcnnEncoder()
    print(modelaaa(btensor).size())

    exit()

    def parameter_number(__model):
        return sum(p.numel() for p in __model.parameters() if p.requires_grad)

    sys.path.append("..")
    import time

    device = torch.device('cuda:0')
    points = torch.randn(8, 1024, 3).to(device)
    model = DGCNN().to(device)

    start = time.time()
    out = model(points)

    print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    print("Input size: {}".format(points.size()))
    print("Out   size: {}".format(out.size()))


if __name__ == "__main__":
    test()

