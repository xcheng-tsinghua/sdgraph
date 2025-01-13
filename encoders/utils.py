import torch
import torch.nn as nn
import torch.nn.functional as F


class full_connected_conv2d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        self.final_proc = final_proc
        if drop_rate == 0:
            self.is_drop = False
        else:
            self.is_drop = True

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm2d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(nn.Dropout2d(drop_rate))

        self.outlayer = nn.Conv2d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm2d(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = nn.Dropout2d(drop_rate)

    def forward(self, embeddings):
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            if self.is_drop:
                fea = dp(at(bn(fc(fea))))
            else:
                fea = at(bn(fc(fea)))
            # fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


class full_connected_conv1d(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        self.final_proc = final_proc
        if drop_rate == 0:
            self.is_drop = False
        else:
            self.is_drop = True

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(nn.Dropout1d(drop_rate))

        self.outlayer = nn.Conv1d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm1d(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = nn.Dropout1d(drop_rate)

    def forward(self, embeddings):
        '''
        :param embeddings: [bs, fea_in, n_points]
        :return: [bs, fea_out, n_points]
        '''
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            if self.is_drop:
                fea = dp(at(bn(fc(fea))))
            else:
                fea = at(bn(fc(fea)))
            # fea = drop(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


class full_connected(nn.Module):
    def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
        '''
        构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param drop_rate: dropout 概率
        '''
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()
        self.n_layers = len(channels)

        self.final_proc = final_proc
        if drop_rate == 0:
            self.is_drop = False
        else:
            self.is_drop = True

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(nn.Dropout(drop_rate))

        self.outlayer = nn.Linear(channels[-2], channels[-1], bias=bias)

        self.outbn = nn.BatchNorm1d(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = nn.Dropout1d(drop_rate)

    def forward(self, embeddings):
        '''
        :param embeddings: [bs, fea_in, n_points]
        :return: [bs, fea_out, n_points]
        '''
        fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            if self.is_drop:
                fea = dp(at(bn(fc(fea))))
            else:
                fea = at(bn(fc(fea)))
            # fea = dp(F.relu(bn(fc(fea))))

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


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


def fps_2d(xyz, n_samples):
    """
    最远采样法进行采样，返回采样点的索引
    Input:
        xyz: pointcloud data, [batch_size, n_points_all, 3]
        n_samples: number of samples
    Return:
        centroids: sampled pointcloud index, [batch_size, n_samples]
    """
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    # 生成 B 行，n_samples 列的全为零的矩阵
    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)

    # 生成 B 行，N 列的矩阵，每个元素为 1e10
    distance = torch.ones(B, N).to(device) * 1e10

    # 生成随机整数tensor，整数范围在[0，N)之间，包含0不包含N，矩阵各维度长度必须用元组传入，因此写成(B,)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]

    return centroids


def knn(vertices, k: int):
    """
    获取每个点最近的k个点的索引
    不包含自身

    vertices: [bs, npoint, 3]
    Return: (bs, npoint, k)
    """
    device = vertices.device
    bs, v, _ = vertices.size()
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)

    neighbor_index = torch.topk(distance, k=k+1, dim=-1, largest=False)[1]

    neighbor_index = neighbor_index[:, :, 1:].to(device)

    return neighbor_index


def index_points(points, idx):
    """
    索引数据时，相当于从[B, N, C]张量中，找到第二维度S个索引对应的数据
    输入[B, N, C], [B, S](int)
    输出[B, S, C]

    索引点时，相当于从[bs, npoint, 3]中，找到每个点的对应的k个点的坐标
    输入[bs, npoint, 3], [bs, npoint, k](int)
    输出[bs, npoint, k, 3]

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


def index_vals(vals, inds):
    '''
    将索引替换为对应的值
    :param vals: size([bs, n_item, n_channel])
    :param inds: size([bs, n_item, n_vals])(int， 索引矩阵，从vals里找到对应的数据填进去)
    :return: size([bs, n_item, n_vals])
    '''
    bs, n_item, n_vals = inds.size()

    # 生成0维度索引
    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)

    # 生成1维度索引
    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)

    return vals[batch_indices, channel_indices, inds]


if __name__ == '__main__':
    pass
