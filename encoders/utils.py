import torch
import torch.nn as nn
import os
import shutil
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

from data_utils.sketch_utils import save_confusion_mat


class FullConnectedConvXd(nn.Module):
    def __init__(self, dimension: int, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
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
            self.linear_layers.append(eval(f'nn.Conv{dimension}d(channels[i], channels[i + 1], 1, bias=bias)'))
            self.batch_normals.append(eval(f'nn.BatchNorm{dimension}d(channels[i + 1])'))
            self.activates.append(activate_func())
            self.drop_outs.append(eval(f'nn.Dropout{dimension}d(drop_rate)'))

        self.outlayer = eval(f'nn.Conv{dimension}d(channels[-2], channels[-1], 1, bias=bias)')

        self.outbn = eval(f'nn.BatchNorm{dimension}d(channels[-1])')
        self.outat = activate_func()
        self.outdp = eval(f'nn.Dropout{dimension}d(drop_rate)')

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

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


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
            self.activates.append(activate_func())
            self.drop_outs.append(nn.Dropout2d(drop_rate))

        self.outlayer = nn.Conv2d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm2d(channels[-1])
        self.outat = activate_func()
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
            self.activates.append(activate_func())
            self.drop_outs.append(nn.Dropout1d(drop_rate))

        self.outlayer = nn.Conv1d(channels[-2], channels[-1], 1, bias=bias)

        self.outbn = nn.BatchNorm1d(channels[-1])
        self.outat = activate_func()
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
            self.activates.append(activate_func())
            self.drop_outs.append(nn.Dropout(drop_rate))

        self.outlayer = nn.Linear(channels[-2], channels[-1], bias=bias)

        self.outbn = nn.BatchNorm1d(channels[-1])
        self.outat = activate_func()
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


def activate_func():
    """
    控制激活函数
    :return:
    """
    return nn.LeakyReLU(negative_slope=0.2)
    # return nn.ReLU()
    # return nn.GELU()
    # return nn.SiLU()


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
    # 即生成初始点的随机索引
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


def fps(xyz, n_samples):
    """
    最远采样法进行采样，返回采样点的索引
    Input:
        xyz: pointcloud data, [batch_size, n_points_all, X]
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
    # 即生成初始点的随机索引
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
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
    将索引值替换为对应的数值
    :param points: [B, N, C] (维度数必须为3)
    :param idx: [A, B, C, D, ..., X]
    :return: [A, B, C, D, ..., X, C]

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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def clear_log(folder_path, k=5):
    """
    遍历文件夹内的所有 .txt 文件，删除行数小于 k 的文件。

    :param folder_path: 要处理的文件夹路径
    :param k: 行数阈值，小于 k 的文件会被删除
    """
    os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查是否为 .txt 文件
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # 统计文件的行数
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    num_lines = len(lines)

                # 如果行数小于 k，则删除文件
                if num_lines < k:
                    print(f"Deleting file: {file_path} (contains {num_lines} lines)")
                    os.remove(file_path)
            except Exception as e:
                # 捕获读取文件时的错误（如编码问题等）
                print(f"Error reading file {file_path}: {e}")


def clear_confusion(root_dir='./data_utils/confusion', k=5):
    """
    遍历 root_dir 中的文件夹，删除文件数小于 k 的文件夹。

    :param root_dir: 根目录
    :param k: 文件数的阈值，小于 k 的文件夹会被删除
    """
    for foldername, subfolders, filenames in os.walk(root_dir, topdown=False):
        # 遍历每个文件夹
        num_files = len(filenames)
        if num_files < k:
            # 如果文件数小于 k，则删除整个文件夹
            print(f"Deleting folder: {foldername} (contains {num_files} files)")
            shutil.rmtree(foldername)


def all_metric_cls(all_preds: list, all_labels: list, confusion_dir: str=''):
    """
    计算分类评价指标：Acc.instance, Acc.class, F1-score, mAP
    :param all_preds: [item0, item1, ...], item: [bs, n_classes]
    :param all_labels: [item0, item1, ...], item: [bs, ]， 其中必须保存整形数据
    :param confusion_dir: 保存 confusion matrix 的路径，为空则不保存
    :return: Acc.instance, Acc.class, F1-score-macro, F1-score-weighted, mAP
    """
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [n_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [n_samples]
    n_samples, n_classes = all_preds.shape

    # 确保all_labels中保存的为整形数据
    if not np.issubdtype(all_labels.dtype, np.integer):
        raise TypeError('all_labels 中保存了非整形数据')

    # ---------- 计算 Acc.Instance ----------
    pred_choice = np.argmax(all_preds, axis=1)  # -> [n_samples, ]
    correct = np.equal(pred_choice, all_labels).sum()
    acc_ins = correct / n_samples

    # ---------- 计算 Acc.class ----------
    acc_cls = []
    for class_idx in range(n_classes):
        class_mask = (all_labels == class_idx)
        if np.sum(class_mask) == 0:
            continue
        cls_acc_sig = np.mean(pred_choice[class_mask] == all_labels[class_mask])
        acc_cls.append(cls_acc_sig)
    acc_cls = np.mean(acc_cls)

    # ---------- 计算 F1-score ----------
    f1_m = f1_score(all_labels, pred_choice, average='macro')
    f1_w = f1_score(all_labels, pred_choice, average='weighted')

    # ---------- 计算 mAP ----------
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(n_classes))
    ap_sig = []
    # 计算单个类别的 ap
    for i in range(n_classes):
        ap = average_precision_score(all_labels_one_hot[:, i], all_preds[:, i])
        ap_sig.append(ap)

    mAP = np.mean(ap_sig)

    # ---------- 保存 confusion matrix ----------
    if confusion_dir != '':
        save_confusion_mat(pred_choice.tolist(), all_labels.tolist(), confusion_dir)

    return acc_ins, acc_cls, f1_m, f1_w, mAP


if __name__ == '__main__':

    import global_defs
    from matplotlib import pyplot as plt

    def vis_sketch_unified(root, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False):
        """
        显示笔划与笔划点归一化后的草图
        """
        # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
        sketch_data = np.loadtxt(root, delimiter=',')

        # 2D coordinates
        coordinates = sketch_data[:, :2]

        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        coordinates = torch.from_numpy(coordinates)
        coordinates = coordinates.view(n_stroke, n_stk_pnt, 2)

        coordinates = coordinates.unsqueeze(0).repeat(5, 1, 1, 1)

        coordinates = coordinates.view(5, n_stroke, n_stk_pnt * 2)
        idxs = torch.randint(0, n_stroke, (10, )).unsqueeze(0).repeat(5, 1)

        print(idxs[0, :])

        coordinates = index_points(coordinates, idxs)
        coordinates = coordinates.view(5, 10, n_stk_pnt, 2)
        coordinates = coordinates[0, :, :, :]


        for i in range(10):
            plt.plot(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

            if show_dot:
                plt.scatter(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

        # plt.axis('off')
        plt.show()


    vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\21.txt')





    pass
