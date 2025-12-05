import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    返回 points 中 索引 idx 对应的点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, channel_coor=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(channel_coor, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", n_neighbor: int):
    """
    Return: (bs, vertice_num, n_neighbor)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=n_neighbor + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, n_neighbor)"):
    """
    Return: (bs, vertice_num, n_neighbor, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, n_neighbor)"):
    """
    获取每个点到其邻近点的向量，会单位化
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    return neighbor_direction_norm


class ConvSurface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, n_support, channel_coor=3):
        super().__init__()
        self.kernel_num = kernel_num
        self.n_support = n_support

        self.relu = nn.ReLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(channel_coor, n_support * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.n_support * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, n_neighbor)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, n_neighbor = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)  # 获得中心点到边缘点的向量 [bs, n_pnt, n_neighobr_num, channel_coor]
        support_direction_norm = F.normalize(self.directions, dim=0)  # (channel_coor, s * k)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, n_neighbor, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, n_neighbor, self.n_support, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, n_support, kernel_num)
        feature = torch.sum(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, n_support, channel_coor=3):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_support = n_support

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (n_support + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((n_support + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(channel_coor, n_support * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.n_support + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, n_neighbor = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, n_neighbor, n_support * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, n_neighbor, -1)
        # (bs, vertice_num, n_neighbor, n_support * out_channel)

        feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (n_support + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, n_support * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support,
                                            neighbor_index)  # (bs, vertice_num, n_neighbor, n_support * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, n_neighbor, n_support * out_channel)
        activation_support = activation_support.view(bs, vertice_num, n_neighbor, self.n_support, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, n_support, out_channel)
        activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature_fuse


class AttnGCN3D(nn.Module):
    """
    多层 3DGcn 特征编码器
    输入: xyz [bs, 3, N]
    输出: fea [bs, channel_out, N]
    """
    def __init__(self, channel_coor=3, channel_fea=0, channel_out=128, n_neighbor=20, attn_k=16, n_support=1):
        """
        注意力机制加 GCN3d 实现特征提取
        :param channel_coor:
        :param channel_fea:
        :param channel_out:
        :param n_neighbor:
        :param attn_k: 只注意最近的前 k 个节点
        :param n_support:
        """
        super().__init__()
        self.n_neighbor = n_neighbor
        self.channel_out = channel_out
        self.attn_k = attn_k
        self.channel_fea = channel_fea

        # 4 层卷积结构
        self.conv0 = ConvSurface(kernel_num=128, n_support=n_support, channel_coor=channel_coor)
        self.conv1 = ConvLayer(128+channel_fea, 128, n_support=n_support, channel_coor=channel_coor)
        self.conv2 = ConvLayer(128, 256, n_support=n_support, channel_coor=channel_coor)
        self.conv3 = ConvLayer(256, channel_out, n_support=n_support, channel_coor=channel_coor)

        # 激活函数
        self.act = nn.ReLU(inplace=True)
        self.attention = TransformerBlock(
            d_points=channel_out,
            d_model=channel_out,
            k=attn_k,
            channel_coor=channel_coor
        )

    def forward(self, xyz, fea=None):
        """
        xyz: [bs, 3, N]
        fea: [bs, channel_fea, N]
        return: [bs, channel_out, N]
        """
        assert fea is None and self.channel_fea == 0 or fea is not None and fea.size(1) == self.channel_fea, ValueError('error fea and channel correspondance')

        bs, _, N = xyz.shape

        # GCN3D 的输入格式：[bs, N, 3]
        vertices = xyz.permute(0, 2, 1)

        # 邻域 [bs, N, K]
        neighbor_index = get_neighbor_index(vertices, self.n_neighbor)

        # 层层卷积
        f0 = self.act(self.conv0(neighbor_index, vertices))        # [bs, N,128]

        if fea is not None:
            fea = fea.permute(0, 2, 1)
            f0 = torch.cat([f0, fea], dim=2)

        f1 = self.act(self.conv1(neighbor_index, vertices, f0))    # [bs, N,128]
        f2 = self.act(self.conv2(neighbor_index, vertices, f1))    # [bs, N,256]
        f3 = self.act(self.conv3(neighbor_index, vertices, f2))    # [bs, N,channel_out]
        xyz_forward = vertices.clone()

        f_attn, _ = self.attention(xyz_forward, f3)

        # 输出转成 [bs, channel_out, N]
        return f_attn.permute(0, 2, 1)


if __name__ == "__main__":
    input_data = torch.randn(32, 2, 2000)
    input_fea = torch.randn(32, 30, 2000)
    model = AttnGCN3D(channel_coor=2, channel_fea=30)
    output = model(input_data, input_fea)
    print(output.shape)


