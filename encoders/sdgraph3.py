"""
笔划均匀化后带mask版本，支持不同长度的笔划和不同笔划数的草图
2025.4.4 测试暂未对扩散模型进行修改
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from encoders.utils import full_connected, full_connected_conv1d, full_connected_conv2d, activate_func
from einops import rearrange
import math
import matplotlib.pyplot as plt

from encoders import Dgcnn
from encoders import dgcnn_orig
import global_defs


def has_invalid_val(x):
    """
    判断Tensor中是否存在无效数值
    :param x:
    :return:
    """
    # 是否存在 nan
    has_nan = torch.isnan(x).any()

    # 是否存在 -nan
    has_neg_inf = (x == float('-inf')).any()  # 检查是否存在 -∞

    # 是否存在 nan
    has_inf = (x == float('inf')).any()  # 检查是否存在 -∞

    return has_nan or has_neg_inf or has_inf


def is_dimension_nan_consistent(tensor: torch.Tensor, dim: int = 1) -> bool:
    """
    检查 tensor 在固定除 dim 外其他维度时，
    对于每个切片沿 dim 的元素，要么全部为 NaN，要么全部不为 NaN。

    参数：
      tensor (torch.Tensor): 输入的多维 tensor。
      dim (int): 指定的维度，对该维度上的元素进行一致性检查。

    返回：
      bool: 如果 tensor 所有切片均符合要求，返回 True；否则返回 False。
    """
    # 获取 tensor 中每个元素是否为 NaN 的布尔掩码
    nan_mask = tensor.isnan()
    inf_mask = tensor.isinf()

    nan_mask = nan_mask.logical_or(inf_mask)

    # 沿指定维度，计算每个切片是否全部为 NaN
    all_true = nan_mask.all(dim=dim)
    # 计算每个切片是否存在至少一个 NaN
    all_false = (~nan_mask).all(dim=dim)

    # 对于每个切片，如果所有元素都是 NaN，all_nan 为 True；
    # 如果没有一个 NaN，any_nan 为 False，则 ~any_nan 为 True。
    # 因此，切片符合条件当且仅当：全部为 NaN 或者全部不为 NaN
    consistent = torch.logical_or(all_true, all_false)

    # 如果所有切片均符合条件，则返回 True，否则返回 False
    return consistent.all().item()


def process_nan_2d(x: torch.Tensor):

    # if mask is not None:
    #     x = x.masked_fill(mask, float('nan'))

    # 1. 沿着批量维度（dim=0）计算每个对应位置的有效值（非 nan）平均值
    nan_mean = x.nanmean(dim=[0, 2, 3], keepdim=True)  # [1, channel, 1, 1]

    # 2. 将 tensor 中的 nan 替换为对应位置的 nan_mean
    # 首先记录 nan 的 mask
    mask = torch.isnan(x)

    # 生成与 x 同形状的 nan_mean 版本，用 unsqueeze 和 expand 将 nan_mean 扩展到批次维度
    nan_mean_expanded = nan_mean.expand_as(x)

    # x[mask] = nan_mean_expanded[mask]
    x = torch.where(mask, nan_mean_expanded, x)

    return x, mask


def process_nan_1d(x: torch.Tensor):
    # 1. 沿着批量维度（dim=0）计算每个对应位置的有效值（非 nan）平均值
    nan_mean = x.nanmean(dim=[0, 2], keepdim=True)  # [1, channel, 1, 1]

    # 2. 将 tensor 中的 nan 替换为对应位置的 nan_mean
    # 首先记录 nan 的 mask
    mask = torch.isnan(x)

    # 生成与 x 同形状的 nan_mean 版本，用 unsqueeze 和 expand 将 nan_mean 扩展到批次维度
    nan_mean_expanded = nan_mean.expand_as(x)

    # x[mask] = nan_mean_expanded[mask]
    x = torch.where(mask, nan_mean_expanded, x)

    return x, mask


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
        # if drop_rate == 0:
        #     self.is_drop = False
        # else:
        #     self.is_drop = True

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm2d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(nn.Dropout2d(drop_rate))

        self.outlayer = nn.Conv2d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm2d(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = nn.Dropout2d(drop_rate)

    def forward(self, fea: torch.Tensor):
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :param fea: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
        # fea = embeddings
        for i in range(self.n_layers - 2):

            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            # 先将nan替换为0，防止nan对反向传播造成影响
            mask = fea.isnan()
            fea = fea.masked_fill(mask, 0)

            fea = fc(fea)

            # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
            fea = fea.masked_fill(mask[:, 0, :, :].unsqueeze(1), float('nan'))

            # 处理前先把nan替换为对应的avg，防止对bn产生影响
            fea, mask = process_nan_2d(fea)

            fea = bn(fea)
            fea = at(fea)
            fea = dp(fea)

            # 再将对应位置替换为nan
            fea = fea.masked_fill(mask, float('nan'))

            # fea = dp(at(bn(fc(fea))))

        # 先将nan替换为0，防止nan对反向传播造成影响
        mask = fea.isnan()
        fea = fea.masked_fill(mask, 0)

        fea = self.outlayer(fea)

        # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
        fea = fea.masked_fill(mask[:, 0, :, :].unsqueeze(1), float('nan'))

        if self.final_proc:
            # 处理前先把nan替换为对应的avg
            fea, mask = process_nan_2d(fea)

            fea = self.outbn(fea)
            fea = self.outat(fea)
            fea = self.outdp(fea)

            # 再将对应位置替换为nan
            fea = fea.masked_fill(mask, float('nan'))
            # fea[mask] = float('nan')

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
        # if drop_rate == 0:
        #     self.is_drop = False
        # else:
        #     self.is_drop = True

        for i in range(self.n_layers - 2):
            self.linear_layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=bias))
            self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(nn.Dropout1d(drop_rate))

        self.outlayer = nn.Conv1d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm1d(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = nn.Dropout1d(drop_rate)

    def forward(self, fea):
        '''
        :param embeddings: [bs, fea_in, n_points]
        :param fea: [bs, fea_in, n_points]
        :return: [bs, fea_out, n_points]
        '''
        # fea = embeddings
        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            # 先将nan替换为0，防止nan对反向传播造成影响
            mask = fea.isnan()
            fea = fea.masked_fill(mask, 0)

            fea = fc(fea)

            # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
            fea = fea.masked_fill(mask[:, 0, :].unsqueeze(1), float('nan'))

            # 处理前先把nan替换为对应的avg
            fea, mask = process_nan_1d(fea)

            fea = bn(fea)
            fea = at(fea)
            fea = dp(fea)

            # 再将对应位置替换为nan
            # fea[mask] = float('nan')
            fea = fea.masked_fill(mask, float('nan'))

        # 先将nan替换为0，防止nan对反向传播造成影响
        mask = fea.isnan()
        fea = fea.masked_fill(mask, 0)

        fea = self.outlayer(fea)

        # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
        fea = fea.masked_fill(mask[:, 0, :].unsqueeze(1), float('nan'))

        if self.final_proc:
            # 处理前先把nan替换为对应的avg
            fea, mask = process_nan_1d(fea)

            fea = self.outbn(fea)
            fea = self.outat(fea)
            fea = self.outdp(fea)

            # 再将对应位置替换为nan
            # fea[mask] = float('nan')
            fea = fea.masked_fill(mask, float('nan'))

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

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


def show_tensor_map(data: torch.Tensor, title=None):

    if len(data.size()) == 1:
        data = data.unsqueeze(1).repeat(1, int(0.2 * data.size(0)))

    # 如果数据在 GPU 上，请先调用 .cpu() 转移到 CPU 后再操作
    # 先将 nan 转换为 inf
    data = data.detach().cpu()

    # data.nan_to_num_(float('-inf'))

    data_np = data.numpy()

    # 将 inf 或 -inf 转换为 nan，这样可以统一处理无效值
    data_np[np.isinf(data_np)] = np.nan

    # 利用 mask 将 nan 无效值屏蔽
    masked_data = np.ma.masked_invalid(data_np)

    # 复制一个 colormap，并设置 masked 值的颜色为黑色
    cmap = plt.cm.viridis.copy()  # 也可以选择其它 colormap
    cmap.set_bad(color='black')

    # 绘制矩阵图
    plt.imshow(masked_data, interpolation='none', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('dim2')
    plt.ylabel('dim1')
    # plt.savefig("matrix_with_black_invalid.png")
    plt.show()


def show_tensor_sketch(data: torch.Tensor):
    """
    tensor中可能有nan
    :param data: [2, n_stk, n_stk_pnt]
    :return:
    """
    data = data.detach().cpu()

    filtered_skh = []

    for i in range(data.size(1)):
        c_stk = []
        c_stk_orig = data[:, i, :]  # [2, n_stk_pnt]

        for j in range(c_stk_orig.size(1)):
            c_pnt = c_stk_orig[:, j]

            if not has_invalid_val(c_pnt):
                c_stk.append(c_pnt.numpy())

        if len(c_stk) > 0:
            filtered_skh.append(np.vstack(c_stk))

    for c_stk in filtered_skh:
        plt.plot(c_stk[0, :], -c_stk[1, :])

    plt.show()


def knn(vertices, neighbor_num):
    """
    找到最近的点的索引，不包含自身
    :param vertices: [bs, n_point, 2]
    :param neighbor_num:
    :return: [bs, n_point, k]
    """
    bs, v, _ = vertices.size()

    # # 先将inf转换为nan
    # vertices[torch.isinf(vertices)] = float('nan')

    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)

    # show_tensor_map(inner[0, :, :])

    # has_neg_inf = (inner == float('-inf')).any()  # 检查是否存在 -∞
    # has_inf = (inner == float('inf')).any()  # 检查是否存在 -∞

    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)  ，可能存在 inf，需要将其转换为 -inf

    # show_tensor_map(quadratic[0, :])

    # has_neg_inf = (quadratic == float('-inf')).any()  # 检查是否存在 -∞
    # has_inf = (quadratic == float('inf')).any()  # 检查是否存在 -∞

    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    # distance.nan_to_num_(float('-inf'))

    # has_neg_inf = (distance == float('-inf')).any()  # 检查是否存在 -∞
    # has_inf = (distance == float('inf')).any()  # 检查是否存在 -∞
    # has_nan = torch.isnan(distance).any()

    # show_tensor_map(distance[0, :, :])

    distance.nan_to_num_(float('inf'))

    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]

    return neighbor_index


def index_points(points, idx):
    """
    将索引值替换为对应的数值
    :param points: [B, N, C] (维度数必须为3)
    :param idx: [A, B, C, D, ..., X]
    :return: [A, B, C, D, ..., X, C]
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


def get_graph_feature(x, k=20):
    """
    找到x中每个点附近的k个点，然后计算中心点到附近点的向量，然后拼接在一起
    :param x: [bs, channel, n_point]
    :param k:
    :return: [bs, channel, n_point, n_near]
    """
    x = x.permute(0, 2, 1)  # -> [bs, n_point, channel]

    # step1: 通过knn计算附近点坐标
    knn_idx = knn(x, k)  # (batch_size, num_points, k)

    # np.savetxt(r'C:\Users\ChengXi\Desktop\sketches\dddl.txt', knn_idx[0, :, :].cpu().numpy(), fmt='%d')

    # step2: 找到附近点
    neighbors = index_points(x, knn_idx)  # -> [bs, n_point, k, channel]

    # step3: 计算向量
    cen_to_nei = neighbors - x.unsqueeze(2)

    # step4: 拼接特征
    feature = torch.cat([cen_to_nei, x.unsqueeze(2).repeat(1, 1, k, 1)], dim=3)  # -> [bs, n_point, k, channel]
    feature = feature.permute(0, 3, 1, 2)  # -> [bs, channel, n_point, k]

    return feature


class Row2DS_Dsample(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super().__init__()

        self.fc = nn.Conv2d(
                in_channels=dim_in,  # 输入通道数 (RGB)
                out_channels=dim_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
        )
        self.bn = nn.BatchNorm2d(dim_in)
        self.at = nn.LeakyReLU(negative_slope=0.2)
        self.dp = nn.Dropout2d(dropout)

    def forward(self, fea: torch.Tensor):
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :param fea: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
        # fea = embeddings

        # 先将nan替换为0，防止nan对反向传播造成影响
        mask = fea.isnan()
        fea = fea.masked_fill(mask, 0)

        fea = self.fc(fea)

        # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
        fea = fea.masked_fill(mask[:, 0, :, ::2].unsqueeze(1), float('nan'))

        # 处理前先把nan替换为对应的avg，防止对bn产生影响
        fea, mask = process_nan_2d(fea)

        fea = self.bn(fea)
        fea = self.at(fea)
        fea = self.dp(fea)

        # 再将对应位置替换为nan
        fea = fea.masked_fill(mask, float('nan'))

        return fea


class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=dim_in,  # 输入通道数 (RGB)
        #         out_channels=dim_out,  # 输出通道数 (保持通道数不变)
        #         kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
        #         stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
        #         padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
        #     ),
        #     nn.BatchNorm2d(dim_out),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout2d(dropout)
        # )
        self.encoder = Row2DS_Dsample(dim_in, dim_out, dropout)

    def forward(self, dense_fea: torch.Tensor):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param mask: [bs, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt // 2], [bs, n_stk, n_stk_pnt // 2]
        """
        # 将mask对应无效位置置为零，防止对卷积产生影响
        # dense_fea = dense_fea.masked_fill(~mask.unsqueeze(1), 0)

        bs, emb, n_stk, n_stk_pnt = dense_fea.size()
        dense_fea = self.encoder(dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt // 2]
        assert is_dimension_nan_consistent(dense_fea, 1)

        # 获取下采样后的mask
        # sampled_mask = mask[:, :, ::2]  # -> [bs, n_stk, n_stk_pnt // 2]

        return dense_fea


class UpSample(nn.Module):
    """
    对sdgraph同时进行上采样
    上采样后笔划数及笔划上的点数均变为原来的2倍
    """
    def __init__(self, dim_in, dim_out, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim_in,  # 输入通道数
                out_channels=dim_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_pnt]
        """
        bs, emb, n_pnt = dense_fea.size()
        assert n_pnt == self.n_stk * self.n_stk_pnt

        dense_fea = dense_fea.view(bs, emb, self.n_stk, self.n_stk_pnt)
        dense_fea = self.conv(dense_fea)
        dense_fea = dense_fea.view(bs, dense_fea.size(1), (self.n_stk * self.n_stk_pnt) * 2)

        return dense_fea


class SinusoidalPosEmb(nn.Module):
    """
    将时间步t转化为embedding
    """
    def __init__(self, dim, theta):  # 256， 10000
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        :param x: 时间步 [bs, ]
        :return:
        """
        device = x.device
        half_dim = self.dim // 2  # 128
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SparseToDense(nn.Module):
    """
    将sgraph转移到dgraph
    直接拼接到该笔划对应的点
    """
    def __init__(self, n_stk, n_stk_pnt):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # TODO: 使用GCNEncoder进行更新，再拼接，需要对顺序无关
        # self.sparse_to_dense = nn.Sequential(
        #     nn.Conv2d(in_channels=sparse_in, out_channels=sparse_in, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(sparse),
        #     activate_func(),
        #     nn.Dropout2d(dropout),
        # )

    def forward(self, sparse_fea, dense_fea) -> torch.Tensor:
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :param mask: [bs, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        dense_feas_from_sparse = sparse_fea.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)

        # show_tensor_map(sparse_fea[0, :, :])

        # show_tensor_map(dense_fea[0, :, :, :].max(0)[0])

        mask = dense_fea.isnan()  # -> [bs, emb, n_stk, n_stk_pnt]

        # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)

        # show_tensor_map(union_dense[0, :, :, :].max(0)[0])

        # 由于拼接后，每个笔划上每个点都会与对应的笔划特征融合，某些笔划中一部分是nan，一部分时有效点，因此会导致无效点的nan上拼接有笔划特征，需要将其全部转换为nan
        union_dense = union_dense.masked_fill(mask[:, 0, :, :].unsqueeze(1), float('nan'))

        # show_tensor_map(union_dense[0, :, :, :].max(0)[0])
        assert is_dimension_nan_consistent(union_dense, 1)

        return union_dense


class Row2DS(nn.Module):
    def __init__(self, dim_in, dropout):
        super().__init__()

        self.fc = nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=(1, 3))
        self.bn = nn.BatchNorm2d(dim_in)
        self.at = nn.LeakyReLU(negative_slope=0.2)
        self.dp = nn.Dropout2d(dropout)

    def forward(self, fea: torch.Tensor):
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :param fea: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
        # fea = embeddings

        # 先将nan替换为0，防止nan对反向传播造成影响
        mask = fea.isnan()
        fea = fea.masked_fill(mask, 0)

        fea = self.fc(fea)

        # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
        fea = fea.masked_fill(mask[:, 0, :, 2:].unsqueeze(1), float('nan'))

        # 处理前先把nan替换为对应的avg，防止对bn产生影响
        fea, mask = process_nan_2d(fea)

        fea = self.bn(fea)
        fea = self.at(fea)
        fea = self.dp(fea)

        # 再将对应位置替换为nan
        fea = fea.masked_fill(mask, float('nan'))

        return fea


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    直接最大池化到一个特征，然后拼接
    """
    def __init__(self, dense_in, n_stk, n_stk_pnt, dropout=0.4):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        # self.dense_to_sparse = nn.Sequential(
        #     nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(dense_in),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout2d(dropout),
        # )
        self.encoder = Row2DS(dense_in, dropout)

    def forward(self, sparse_fea, dense_fea) -> torch.Tensor:
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param mask: [bs, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk]
        """
        # show_tensor_map(dense_fea[0, :, :, :].max(0)[0], 'orig dn fea')

        # -> [bs, emb, n_stk, n_stk_pnt - 2]
        sparse_feas_from_dense = self.encoder(dense_fea)

        # show_tensor_map(sparse_feas_from_dense[0, :, :, :].max(0)[0], 'encoded dn fea')

        # 将dense graph对应的无效位置置为'-inf'，从而在max时忽略这些位置的数据
        mask = sparse_feas_from_dense.isnan()  # -> [bs, emb, n_stk, n_stk_pnt - 2]

        # show_tensor_map(mask[0, :, :, :].max(0)[0], 'nan mask of encoded dn fea')

        sparse_feas_from_dense = sparse_feas_from_dense.masked_fill(mask, float('-inf'))

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = sparse_feas_from_dense.max(3)[0]  # -> [bs, emb, n_stk]
        assert sparse_feas_from_dense.size(2) == self.n_stk
        assert is_dimension_nan_consistent(sparse_feas_from_dense, 1)

        # 再将对应无效位置替换为nan，因为模块输入时统一以nan表示无效位置
        # 因为指定的是nan的位置，因此是min而不是max
        mask = mask.min(3)[0].bool()  # -> [bs, emb, n_stk]
        sparse_feas_from_dense = sparse_feas_from_dense.masked_fill(mask, float('nan'))  # -> [bs, emb, n_stk]

        # show_tensor_map(sparse_fea[0], 'sp fea')
        # show_tensor_map(sparse_feas_from_dense[0], 'dn fea')

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)

        # assert is_dimension_nan_consistent(union_sparse, 1)
        if not is_dimension_nan_consistent(union_sparse, 1):
            assert is_dimension_nan_consistent(sparse_fea)
            bs = sparse_fea.size(0)

            for i in range(bs):

                show_tensor_map(sparse_fea[i], 'sp fea')

                show_tensor_map(sparse_feas_from_dense[i], 'sp fea from dense')

                show_tensor_map(union_sparse[i], 'union sp fea')

                asasa = 0

        # show_tensor_map(union_sparse[0], 'union sp fea')

        return union_sparse


class GCNEncoder(nn.Module):
    """
    实际上是 DGCNN Encoder
    """
    def __init__(self, emb_in, emb_out, n_near=10):
        super().__init__()
        self.n_near = n_near

        emb_inc = (emb_out / (4 * emb_in)) ** 0.25
        emb_l1_0 = emb_in * 2
        emb_l1_1 = int(emb_l1_0 * emb_inc)
        emb_l1_2 = int(emb_l1_0 * emb_inc ** 2)

        emb_l2_0 = emb_l1_2 * 2
        emb_l2_1 = int(emb_l2_0 * emb_inc)
        emb_l2_2 = emb_out

        emb_l3_0 = emb_l2_2 + emb_l1_2
        emb_l3_1 = int(((emb_out / emb_l3_0) ** 0.5) * emb_l3_0)
        emb_l3_2 = emb_out

        self.conv1 = full_connected_conv2d([emb_l1_0, emb_l1_1, emb_l1_2],
                                           final_proc=True,
                                           drop_rate=0.0
                                           )
        self.conv2 = full_connected_conv2d([emb_l2_0, emb_l2_1, emb_l2_2],
                                           final_proc=True,
                                           drop_rate=0.0
                                           )

        self.conv3 = full_connected_conv1d([emb_l3_0, emb_l3_1, emb_l3_2],
                                           final_proc=True, drop_rate=0.0
                                           )

    def forward(self, x: torch.Tensor):
        """
        :param x: [bs, channel, n_token]
        :return:
        """
        x = get_graph_feature(x, k=self.n_near)  # -> [bs, 2 * channel, n_token, n_near]
        x = self.conv1(x)

        # 进行max之前先将nan转换为-inf，以防止产生影响
        mask = x.isnan()  # -> [bs, channel, n_token, n_near]
        x = x.masked_fill(mask, float('-inf'))
        x1 = x.max(3)[0]  # -> [bs, channel, n_token]

        # 然后将 -inf 还原为 nan
        x1 = x1.masked_fill(mask[:, :, :, 0], float('nan'))  # -> [bs, channel, n_token]

        assert is_dimension_nan_consistent(x1, 1)

        x = get_graph_feature(x1, k=self.n_near)
        x = self.conv2(x)

        # 进行max之前先将nan转换为-inf，以防止产生影响
        mask = x.isnan()  # -> [bs, channel, n_token, n_near]
        x = x.masked_fill(mask, float('-inf'))
        x2 = x.max(dim=-1, keepdim=False)[0]

        # 然后将 -inf 还原为 nan
        x2 = x2.masked_fill(mask[:, :, :, 0], float('nan'))  # -> [bs, channel, n_token]

        assert is_dimension_nan_consistent(x2, 1)

        # -> [bs, emb, n_token]
        x = torch.cat((x1, x2), dim=1)  # -> [bs, channel, n_token]

        assert is_dimension_nan_consistent(x, 1)

        # -> [bs, emb, n_token]
        x = self.conv3(x)

        assert is_dimension_nan_consistent(x, 1)

        return x


# class GCNEncoderSingle(nn.Module):
#     """
#     实际上是 DGCNN Encoder
#     """
#     def __init__(self, emb_in, emb_out, n_near=10,):
#         super().__init__()
#         self.n_near = n_near
#
#         emb_inc = (emb_out / (2 * emb_in)) ** 0.25
#         emb_l1_0 = emb_in * 2
#         emb_l1_1 = int(emb_l1_0 * emb_inc)
#         emb_l1_2 = int(emb_l1_1 * emb_inc)
#
#         emb_l2_0 = emb_l1_2
#         emb_l2_1 = int(emb_l2_0 * emb_inc)
#         emb_l2_2 = emb_out
#
#         self.conv1 = full_connected_conv2d([emb_l1_0, emb_l1_1, emb_l1_2],
#                                            final_proc=True,
#                                            drop_rate=0.0
#                                            )
#
#         self.conv2 = full_connected_conv1d([emb_l2_0, emb_l2_1, emb_l2_2],
#                                            final_proc=True,
#                                            drop_rate=0.0
#                                            )
#
#     def forward(self, x, mask=None):
#         """
#         :param x: [bs, channel, n_token]
#         :param mask: [bs, n_stk, n_stk_pnt]
#         :return:
#         """
#         # -> [bs, channel, n_points, n_near]
#         # x = get_graph_feature(x, k=self.n_near)
#         x = get_graph_feature(x, 2)
#
#         # mask 之后
#         # mask = mask.view(mask.size(0), -1)  # -> [bs, n_point]
#         # mask = mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(3)  # -> [bs, channel, n_point, 1]
#         # x = x.masked_fill(~mask, 0)
#
#         # has_nan = torch.isnan(x).any()  # 检查是否存在 NaN
#         # has_neg_inf = (x == float('-inf')).any()  # 检查是否存在 -∞
#
#         # x[torch.isinf(x)] = float('nan')
#         # aaal = x[0, :, :, :].max(0)[0]
#         # bbbl = mask[0, :, :].view(-1)
#
#
#         # aaal = x[0, :, :, :].max(2)[0]
#         # bbbl = mask[0, :, :].view(-1)
#
#         # aaal = x.permute(0, 2, 3, 1)  # -> [bs, n_points, n_near, channel]
#         # aaal = aaal.reshape(aaal.size(0), aaal.size(1), -1)  # -> [bs, n_points, n_near * channel]
#         # aaal = aaal[0, :, :]  # -> [n_points, n_near * channel]
#         #
#         # bbbl = mask[0, :, :].view(-1)  # -> [n_points]
#
#         # np.savetxt(r'C:\Users\ChengXi\Desktop\sketches\aaal.txt', aaal.cpu().numpy())
#         # np.savetxt(r'C:\Users\ChengXi\Desktop\sketches\bbbl.txt', bbbl.cpu().numpy(), fmt='%d')
#         # print(aaal)
#         # exit(0)
#
#
#
#         x = self.conv1(x)
#
#         # np.savetxt(r'C:\Users\ChengXi\Desktop\sketches\cccl.txt', x[0, :, :, :].max(0)[0].detach().cpu().numpy())
#
#         # -> [bs, emb, n_token]
#         x = x.max(3)[0]
#
#         # -> [bs, emb, n_token]
#         x = self.conv2(x)
#
#         # mask = mask.view(mask.size(0), -1)  # [bs, n_points]
#
#         # x = x.masked_fill(~mask.unsqueeze(1), 0)
#
#         # has_nan = has_invalid_val(x)  # 检查是否存在 NaN
#
#
#         return x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        """
        :param dim: forward过程中输入x的特征长度
        """
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class TimeMerge(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(time_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim_in, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """
        :param x: [bs, channel, n_node]
        :param scale_shift: [bs, ]
        :return:
        """
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.dropout(self.act(x))
        return x


class Row2D(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()

        mid_dim = int((dim_in * dim_out) ** 0.5)
        channels = [dim_in, mid_dim, dim_out]
        self.n_layers = len(channels)

        for i in range(self.n_layers - 1):
            self.linear_layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=(1, 3)))
            self.batch_normals.append(nn.BatchNorm2d(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))

    def forward(self, fea: torch.Tensor):
        '''
        :param embeddings: [bs, fea_in, n_row, n_col]
        :param fea: [bs, fea_in, n_row, n_col]
        :return: [bs, fea_out, n_row, n_col]
        '''
        # fea = embeddings
        for i in range(self.n_layers - 1):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]

            # 先将nan替换为0，防止nan对反向传播造成影响
            mask = fea.isnan()
            fea = fea.masked_fill(mask, 0)

            fea = fc(fea)

            # 再将nan对应维度替换为nan，防止对后续处理产生影响，因为模块起始统一将无效位置标记为nan
            fea = fea.masked_fill(mask[:, 0, :, 2:].unsqueeze(1), float('nan'))

            # 处理前先把nan替换为对应的avg，防止对bn产生影响
            fea, mask = process_nan_2d(fea)

            fea = bn(fea)
            fea = at(fea)

            # 再将对应位置替换为nan
            fea = fea.masked_fill(mask, float('nan'))

        return fea


class PointToSparse(nn.Module):
    """
    利用点生成 sparse graph
    """

    def __init__(self, point_dim, sparse_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt,
                 with_time=False, time_emb_dim=0):
        super().__init__()

        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        # 将 DGraph 的数据转移到 SGraph
        # 通过该卷积层，向量长度 -4
        # mid_dim = int((point_dim * sparse_out) ** 0.5)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=point_dim, out_channels=mid_dim, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(mid_dim),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     # nn.Dropout2d(dropout),
        #
        #     nn.Conv2d(in_channels=mid_dim, out_channels=sparse_out, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(sparse_out),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     # nn.Dropout2d(dropout),
        # )

        self.encoder = Row2D(point_dim, sparse_out)

        if self.with_time:
            self.time_merge = TimeMerge(sparse_out, sparse_out, time_emb_dim)

    def forward(self, xy: torch.Tensor, time_emb=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param mask: [bs, n_stk, n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk]
        """
        bs, n_stk, n_stk_pnt, channel_xy = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_xy == 2

        # -> [bs, 2, n_stk, n_stk_pnt]
        xy = xy.permute(0, 3, 1, 2)

        # show_tensor_sketch(xy[0, :, :, :])
        # xy.nan_to_num_(float('-inf'))
        # show_tensor_map(xy[0, :, :, :].max(0)[0], 'orig xy')

        # 记录无效位置, -> [bs, 2, n_stk, n_stk_pnt]
        # mask = torch.isnan(xy)

        # 卷积应在mask指定的无效位置补零
        # xy = xy.masked_fill(mask, 0)
        xy = self.encoder(xy)  # -> [bs, emb, n_stk, n_stk_pnt - 4]

        # hasnan = has_invalid_val(xy)

        # show_tensor_map(xy[0, :, :, :].max(0)[0], 'encoded xy')

        assert is_dimension_nan_consistent(xy, 1)

        # 由于卷积后向量长度 -4，因此mask需要从左数4列全部删除
        # mask = mask[:, :, :, 4:]  # -> [bs, 2, n_stk, n_stk_pnt - 4]

        # 将点特征的无效位置置为 float('-inf')，从而在使用max时使得该位置的特征被忽略
        mask = xy.isnan()  # -> [bs, emb, n_stk, n_stk_pnt - 4]

        # show_tensor_map(mask[0, :, :, :].max(0)[0], 'mask of encoded xy')

        xy = xy.masked_fill(mask, float('-inf'))

        # show_tensor_map(xy[0, :, :, :].max(0)[0], 'inf filled encoded xy')

        xy = xy.max(3)[0]  # -> [bs, emb, n_stk]
        assert xy.size(2) == self.n_stk

        # show_tensor_map(xy[0, :, :], 'inf filled encoded stk maxed xy')

        # 可能有些位置的笔划特征为 '-inf'，使其值为nan，保证后续可以正确识别mask
        # 因为这里的mask为inf的位置，而不是有效数值的位置，因此是min
        mask = mask.min(3)[0].bool()  # -> [bs, emb, n_stk]

        # show_tensor_map(mask[0, :, :], 'stk emb mask')

        xy = xy.masked_fill(mask, float('nan'))

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            xy = self.time_merge(xy, time_emb)

        # show_tensor_map(xy[0, :, :], 'inf filled encoded stk maxed and nan filled xy')

        return xy


class PointToDense(nn.Module):
    """
    利用点坐标生成 dense graph
    使用DGCNN直接为每个点生成对应特征
    """
    def __init__(self, point_dim, emb_dim, with_time=False, time_emb_dim=0, n_near=10):
        super().__init__()
        # self.encoder = GCNEncoderSingle(point_dim, emb_dim, n_near)
        self.encoder = GCNEncoder(point_dim, emb_dim, n_near)

        self.with_time = with_time
        if self.with_time:
            self.time_merge = TimeMerge(emb_dim, emb_dim, time_emb_dim)

    def forward(self, xy: torch.Tensor, time_emb=None) -> torch.Tensor:
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param mask: [bs, n_stk, n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        bs, n_stk, n_stk_pnt, channel_xy = xy.size()
        assert n_stk == global_defs.n_stk and n_stk_pnt == global_defs.n_stk_pnt and channel_xy == 2

        xy = xy.view(bs, n_stk * n_stk_pnt, channel_xy)
        xy = xy.permute(0, 2, 1)  # [bs, 2, n_stk * n_stk_pnt]

        dense_emb = self.encoder(xy)  # [bs, emb, n_stk * n_stk_pnt]
        dense_emb = dense_emb.view(bs, dense_emb.size(1), n_stk, n_stk_pnt)  # [bs, emb, n_stk, n_stk_pnt]

        assert is_dimension_nan_consistent(dense_emb, 1)

        # 将mask对应的无效点位置置为0，防止对后续计算产生影响
        # dense_emb = dense_emb.masked_fill(~mask.unsqueeze(1), 0)

        # assert not has_invalid_val(dense_emb)

        # has_nan = has_invalid_val(dense_emb)
        # if has_nan:
        #     raise ValueError('contain nan')

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            dense_emb = self.time_merge(dense_emb, time_emb)

        return dense_emb


class SDGraphEncoder(nn.Module):
    def __init__(self,
                 sparse_in, sparse_out, dense_in, dense_out,  # 输入输出维度
                 n_stk, n_stk_pnt,  # 笔划数，每个笔划中的点数
                 sp_near=4, dn_near=10,  # 更新sdgraph的两个GCN中邻近点数目
                 sample_type='down_sample',  # 采样类型
                 with_time=False, time_emb_dim=0,  # 是否附加时间步
                 dropout=0.4
                 ):
        """
        :param sample_type: [down_sample, up_sample, none]
        """
        super().__init__()
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        self.dense_to_sparse = DenseToSparse(dense_in, n_stk, n_stk_pnt, dropout)  # 这个不能设为零
        self.sparse_to_dense = SparseToDense(n_stk, n_stk_pnt)

        self.sparse_update = GCNEncoder(sparse_in + dense_in, sparse_out, sp_near)
        self.dense_update = GCNEncoder(dense_in + sparse_in, dense_out, dn_near)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(dense_out, dense_out, self.n_stk, self.n_stk_pnt, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)  # 这里dropout不能为零
            self.time_mlp_dn = TimeMerge(dense_out, dense_out, time_emb_dim, dropout)  # 这里dropout不能为零

    def forward(self, sparse_fea, dense_fea, time_emb=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param mask: [bs, n_stk, n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk], [bs, emb, n_stk, n_stk_pnt // 2]
        """
        bs, emb, n_stk, n_stk_pnt = dense_fea.size()
        assert n_stk == self.n_stk == sparse_fea.size(2) and n_stk_pnt == self.n_stk_pnt

        # show_tensor_map(sparse_fea[0, :, :])
        # show_tensor_map(dense_fea[0, :, :, :].max(0)[0])

        # 信息交换
        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)  # -> [bs, emb, n_stk]

        union_dense = self.sparse_to_dense(sparse_fea, dense_fea)  # -> [bs, emb, n_stk, n_stk_pnt]

        # 信息更新
        union_sparse = self.sparse_update(union_sparse)  # -> [bs, emb, n_stk]

        union_dense = self.dense_update(union_dense.view(bs, union_dense.size(1), n_stk * n_stk_pnt))  # -> [bs, emb, n_stk * n_stk_pnt]

        union_dense = union_dense.view(bs, union_dense.size(1), n_stk, n_stk_pnt)  # -> [bs, emb, n_stk, n_stk_pnt]

        # 下采样
        union_dense = self.sample(union_dense)

        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense


class TimeEncode(nn.Module):
    """
    编码时间步
    """
    def __init__(self, channel_time):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(channel_time // 4, theta=10000),
            nn.Linear(channel_time // 4, channel_time // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channel_time // 2, channel_time)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class SDGraphCls(nn.Module):
    def __init__(self, n_class: int, dropout: float = 0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('cls stk alt')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        # 各层特征维度
        # sparse_l0 = 16 + 8
        # sparse_l1 = 64 + 32
        # sparse_l2 = 256 + 128
        #
        # dense_l0 = 16
        # dense_l1 = 64
        # dense_l2 = 256

        sparse_l0 = 8
        sparse_l1 = 16
        sparse_l2 = 32

        dense_l0 = 4
        dense_l1 = 8
        dense_l2 = 16


        # 生成初始 sdgraph
        self.point_to_sparse = PointToSparse(2, sparse_l0)
        self.point_to_dense = PointToDense(2, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  self.n_stk, self.n_stk_pnt,
                                  dropout=dropout
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  self.n_stk, self.n_stk_pnt // 2,
                                  dropout=dropout
                                  )

        # 利用输出特征进行分类
        sparse_glo = sparse_l0 + sparse_l1 + sparse_l2
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / (sparse_glo + dense_glo)) ** (1 / 3)

        out_l0 = sparse_glo + dense_glo
        out_l1 = int(out_l0 * out_inc)
        out_l2 = int(out_l1 * out_inc)
        out_l3 = n_class

        self.linear = full_connected(channels=[out_l0, out_l1, out_l2, out_l3], final_proc=False, drop_rate=dropout)

    def forward(self, xy: torch.Tensor):
        """
        输入某个模块前，无效位置统一设为 nan
        :param xy: [bs, n_stk, n_skt_pnt, 2]  float
        # :param mask: [bs, n_stk, n_skt_pnt]  bool
        :return: [bs, n_classes]
        """
        bs, n_stk, n_stk_pnt, channel_xy = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_xy == 2
        # assert mask.size(1) == self.n_stk and mask.size(2) == self.n_stk_pnt

        # 生成初始 sparse graph
        sparse_graph0 = self.point_to_sparse(xy)  # -> [bs, emb, n_stk]
        assert sparse_graph0.size()[2] == self.n_stk

        assert is_dimension_nan_consistent(sparse_graph0, 1)

        # if sparse_graph0.isnan().any().item():
        #     raise ValueError('nan occurred')

        # 生成初始 dense graph
        dense_graph0 = self.point_to_dense(xy)  # -> [bs, emb, n_stk, n_stk_pnt]
        assert dense_graph0.size()[2] == self.n_stk and dense_graph0.size()[3] == self.n_stk_pnt

        # if dense_graph0.isnan().any().item():
        #     raise ValueError('nan occurred')

        # show_tensor_map(dense_graph0[0, :, :, :].max(0)[0], 'dn0')
        # show_tensor_map(sparse_graph0[0, :, :], 'sp0')

        # 交叉更新数据
        sparse_graph1, dense_graph1 = self.sd1(sparse_graph0, dense_graph0)  # [bs, emb, n_stk], [bs, emb, n_stk, n_stk_pnt // 2]
        sparse_graph2, dense_graph2 = self.sd2(sparse_graph1, dense_graph1)  # [bs, emb, n_stk], [bs, emb, n_stk, n_stk_pnt // 4]

        # show_tensor_map(dense_graph1[0, :, :, :].max(0)[0], 'dn1')
        # show_tensor_map(sparse_graph1[0, :, :], 'sp1')
        #
        # show_tensor_map(dense_graph2[0, :, :, :].max(0)[0], 'dn2')
        # show_tensor_map(sparse_graph2[0, :, :], 'sp2')

        # if dense_graph1.isnan().any().item():
        #     raise ValueError('nan occurred')
        #
        # if dense_graph2.isnan().any().item():
        #     raise ValueError('nan occurred')
        #
        # if sparse_graph1.isnan().any().item():
        #     raise ValueError('nan occurred')
        #
        # if sparse_graph2.isnan().any().item():
        #     raise ValueError('nan occurred')

        # 提取全局特征
        # 将各阶段的 sparse_graph 的nan位置置为 '-inf'，从而在max时忽略这些位置
        s0 = sparse_graph0.nan_to_num(float('-inf'))
        s1 = sparse_graph1.nan_to_num(float('-inf'))
        s2 = sparse_graph2.nan_to_num(float('-inf'))

        sparse_glo0 = s0.max(2)[0]
        sparse_glo1 = s1.max(2)[0]
        sparse_glo2 = s2.max(2)[0]

        d0 = dense_graph0.nan_to_num(float('-inf'))
        d1 = dense_graph1.nan_to_num(float('-inf'))
        d2 = dense_graph2.nan_to_num(float('-inf'))

        dense_glo0 = d0.max(2)[0].max(2)[0]
        dense_glo1 = d1.max(2)[0].max(2)[0]
        dense_glo2 = d2.max(2)[0].max(2)[0]

        all_fea = torch.cat([sparse_glo0, sparse_glo1, sparse_glo2, dense_glo0, dense_glo1, dense_glo2], dim=1)

        assert not has_invalid_val(all_fea)

        # 利用全局特征分类
        cls = self.linear(all_fea)
        cls = F.log_softmax(cls, dim=1)

        return cls


class SDGraphUNet(nn.Module):
    def __init__(self, channel_in, channel_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.0):
        super().__init__()
        print('diff drop 0')

        '''草图参数'''
        self.channel_in = channel_in
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

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
        self.point_to_sparse = PointToSparse(channel_in, sparse_l0, with_time=True, time_emb_dim=time_emb_dim)
        self.point_to_dense = PointToDense(channel_in, dense_l0, with_time=True, time_emb_dim=time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                       self.n_stk, self.n_stk_pnt,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                       self.n_stk, self.n_stk_pnt // 2,
                                       sp_near=2, dn_near=10,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        '''全局特征生成层'''
        global_in = sparse_l2 + dense_l2
        self.global_linear = full_connected(
            channels=[global_in, int((global_in * global_dim) ** 0.5), global_dim],
            final_proc=True, drop_rate=dropout)

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 4,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     n_stk=self.n_stk, n_stk_pnt=self.n_stk_pnt // 2,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        '''最终输出层'''
        final_in = dense_l0 + sparse_l0 + dense_l1 + sparse_l1 + channel_in
        self.final_linear = full_connected_conv1d(
            channels=[final_in, int((channel_out * final_in) ** 0.5), channel_out],
            final_proc=False,
            drop_rate=dropout
        )

    def channels(self):
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
        sparse_graph_up0 = self.point_to_sparse(xy, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_up0 = self.point_to_dense(xy, time_emb)  # -> [bs, emb, n_point]
        assert sparse_graph_up0.size()[2] == self.n_stk and dense_graph_up0.size()[2] == n_point

        '''下采样'''
        sparse_graph_up1, dense_graph_up1 = self.sd_down1(sparse_graph_up0, dense_graph_up0, time_emb)
        sparse_graph_up2, dense_graph_up2 = self.sd_down2(sparse_graph_up1, dense_graph_up1, time_emb)

        '''获取全局特征'''
        sp_up2_glo = sparse_graph_up2.max(2)[0]
        dn_up2_glo = dense_graph_up2.max(2)[0]

        fea_global = torch.cat([sp_up2_glo, dn_up2_glo], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征 (直接拼接在后面)'''
        sparse_fit = fea_global.unsqueeze(2).repeat(1, 1, self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = fea_global.unsqueeze(2).repeat(1, 1, dense_graph_up2.size(2))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2, time_emb)  # -> [bs, sp_l2, n_stk], [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1, time_emb)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = sparse_graph.unsqueeze(3).repeat(1, 1, 1, self.n_stk_pnt)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk, self.n_stk_pnt)
        xy = xy.view(bs, channel_in, self.n_stk, self.n_stk_pnt)

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)
        return noise


class SDGraphClsTest(nn.Module):
    """
    逐模块测试版
    """
    def __init__(self, n_class: int, dropout: float = 0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('cls stk alt')

        self.point_to_dense = PointToDense(2, 16)

        self.conv = GCNEncoder(16, 128, 20)

        dim_mid = int((128 * n_class) ** 0.5)

        self.linear = full_connected(channels=[128, dim_mid, n_class], final_proc=False, drop_rate=dropout)

        # self.conv = dgcnn_orig.DGCNN(n_class)

    def forward(self, xy: torch.Tensor):
        """
        输入某个模块前，无效位置统一设为 nan
        :param xy: [bs, n_stk, n_skt_pnt, 2]  float
        # :param mask: [bs, n_stk, n_skt_pnt]  bool
        :return: [bs, n_classes]
        """
        # bs, n_stk, n_stk_pnt, channel_xy = xy.size()
        # assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_xy == 2
        #
        # xy = xy.view(bs, n_stk * n_stk_pnt, channel_xy)
        # xy = xy.permute(0, 2, 1)

        xy = xy.view(xy.size(0), 2, global_defs.n_stk, global_defs.n_stk_pnt)
        xy = xy.permute(0, 2, 3, 1)  # -> [bs, n_stk, n_stk_pnt, 2]

        xy = self.point_to_dense(xy)
        xy = xy.view(xy.size(0), xy.size(1), -1)

        # -> [bs, fea, n_pnt]
        fea = self.conv(xy)

        # -> [bs, fea]
        fea = fea.max(2)[0]

        # -> [bs, n_classes]
        fea = self.linear(fea)

        fea = F.log_softmax(fea, dim=1)

        return fea


def test():
    #     bs = 3
    #     atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
    #     t1 = torch.randint(0, 1000, (bs,)).long().cuda()
    #
    #     # classifier = SDGraphSeg2(2, 2).cuda()
    #     # cls11 = classifier(atensor, t1)
    #
    #     classifier = SDGraphCls2(10).cuda()
    #     cls11 = classifier(atensor)
    #
    #     print(cls11.size())
    #
    #     print('---------------')
    # bs = 3
    # atensor = torch.rand([bs, 2, global_defs.n_skh_pnt])
    # t1 = torch.randint(0, 1000, (bs,)).long()
    #
    # # classifier = SDGraphSeg2(2, 2)
    # # cls11 = classifier(atensor, t1)
    #
    # classifier = SDGraphCls(10)
    # cls11 = classifier(atensor)
    #
    # print(cls11.size())
    #
    # print('---------------')

    def get_graph(x, k=20):
        """
        找到x中每个点附近的k个点，然后计算中心点到附近点的向量，然后拼接在一起
        :param x: [bs, n_point, channel]
        :param k:
        :return: [bs, channel, n_point, n_near]
        """
        # step1: 通过knn计算附近点坐标
        knn_idx = knn(x, k)  # (batch_size, num_points, k)

        # step2: 找到附近点
        neighbors = index_points(x, knn_idx)  # -> [bs, n_point, k, channel]

        # step3: 计算向量
        cen_to_nei = neighbors - x.unsqueeze(2)

        # step4: 拼接特征
        feature = torch.cat([cen_to_nei, x.unsqueeze(2).repeat(1, 1, k, 1)], dim=3)  # -> [bs, n_point, k, channel]
        feature = feature.permute(0, 3, 1, 2)  # -> [bs, channel, n_point, k]

        return feature

    # 先取一些点
    pnt_list = torch.rand(2, 10, 2)
    pnt_list[:, 9, :] = torch.tensor([float('-inf'), float('-inf')])
    pnt_list[:, 3, :] = torch.tensor([float('-inf'), float('-inf')])
    pnt_list[0, 5, :] = torch.tensor([float('-inf'), float('-inf')])
    # pnt_list = pnt_list.unsqueeze(0)  # [bs, n_pnt, 2]

    pnt_list = get_graph(pnt_list, k=3)
    # pnt_list[torch.isinf(pnt_list)] = float('nan')
    # pnt_list = pnt_list.max(1)[0]

    print(pnt_list)



def test_topk():
    pnt_list = torch.rand(4, 2)
    # pnt_list[9, :] = torch.tensor([float('-inf'), float('-inf')])
    # pnt_list[3, :] = torch.tensor([float('-inf'), float('-inf')])
    pnt_list[1, :] = torch.tensor([float('-inf'), float('-inf')])

    print('points:\n', pnt_list)

    pnt_list[torch.isinf(pnt_list)] = float('nan')
    inner = pnt_list @ pnt_list.transpose(0, 1)  # (v, v)

    print('inner:\n', inner)

    quadratic = torch.sum(pnt_list**2, dim=1)  # (v)

    print('quadratic:\n', quadratic)

    distance = inner * (-2.0) + quadratic.unsqueeze(0) + quadratic.unsqueeze(1)
    distance.nan_to_num_(float('inf'))
    print('distance:\n', distance)

    neighbor_index = torch.topk(distance, k=3, dim=-1, largest=False)[1]


    print('topk:\n', neighbor_index)







    pass

def test_sketch():
    from data_utils.sketch_utils import short_straw_split_sketch

    file_name = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Key\0a4b71aa11ae34effcdc8e78292671a3_2.txt'

    sketch_data = short_straw_split_sketch(file_name, is_show_status=False)

    # 创建 mask 和规则的 sketch
    sketch_mask = torch.zeros(global_defs.n_stk, global_defs.n_stk_pnt, dtype=torch.int)
    sketch_cube = torch.zeros(global_defs.n_stk, global_defs.n_stk_pnt, 2, dtype=torch.float)

    # sketch_cube = torch.full((global_defs.n_stk, global_defs.n_stk_pnt, 2), float('-inf'))
    for i, c_stk in enumerate(sketch_data):
        n_cstk_pnt = len(c_stk)
        sketch_mask[i, :n_cstk_pnt] = 1
        sketch_cube[i, :n_cstk_pnt, :] = torch.from_numpy(c_stk)

    sketch_mask = sketch_mask.bool()

    sketch_cube = sketch_cube.unsqueeze(0).repeat(2, 1, 1, 1)
    sketch_mask = sketch_mask.unsqueeze(0).repeat(2, 1, 1)

    cls_model = SDGraphCls(10)
    res = cls_model(sketch_cube, sketch_mask)


def test_batch_norm():
    # 设置随机种子，保证结果可复现

    torch.manual_seed(1107)

    # 创建一个 4D 张量，形状为 (2, 3, 4, 4)

    x = torch.rand(2, 3, 4)

    # 实例化 BatchNorm2d，通道数为 3，momentum 设置为 1

    m = nn.BatchNorm1d(3, momentum=1)

    y = m(x)

    # 手动计算 BatchNorm2d

    x_mean = x.mean(dim=[0, 2], keepdim=True)  # 按通道计算均值

    x_var = x.var(dim=[0, 2], keepdim=True, unbiased=False)  # 按通道计算方差（无偏）

    eps = m.eps  # 获取 epsilon 值

    y_manual = (x - x_mean) / ((x_var + eps).sqrt())  # 标准化公式

    # 检查两种方法的输出是否一致

    print("使用 BatchNorm2d 的结果：", y)

    print("手动计算的结果：", y_manual)

    print("结果是否一致：", torch.allclose(y, y_manual, atol=1e-6))





if __name__ == '__main__':
    # test()
    # aten = torch.ones(4, 5)
    # masks = torch.tensor([1, 0, 0, 1], dtype=torch.int)
    # aten[masks.unsqueeze(-1).repeat(1, 5) == 0] = float('-inf')
    # print(aten)

    # a = float('-inf')
    # print(a * 0)
    #
    # print('---------------')

    # test()
    # test_topk()
    # test_sketch()
    # test_batch_norm()

    pnt_list = torch.rand(4, 2)
    # pnt_list[9, :] = torch.tensor([float('-inf'), float('-inf')])
    # pnt_list[3, :] = torch.tensor([float('-inf'), float('-inf')])
    # pnt_list[1, :] = torch.tensor([float('inf'), float('nan')])
    pnt_list[1, 0] = float('-inf')
    assasasasadf = is_dimension_nan_consistent(pnt_list, 1)
    print(assasasasadf)
