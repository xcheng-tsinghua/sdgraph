"""
草图数据集加载及转化相关

std 草图：
保存为 .txt 文件，每行记录一个点，数据之间以英文逗号分隔。std草图中不同草图的笔划数及笔划上的点数可能不同。草图质心在(0, 0)，范围在[-1, 1]^2
x, y, s
x, y, s
...
x, y, s

s: 该点的下一个点是否属于当前笔划
s = 1: 该点的下一个点属于当前笔划
s = 0: 该点的下一个点不属于当前笔划

unified_std 草图：
保存方式与 std 草图类似，不同点在于 unified_std 草图中不同草图的笔划数及笔划上的点数相同

"""
import numpy as np
from torch.utils.data import Dataset
import shutil
import torch
import os
from tqdm import tqdm
import re
import warnings

import encoders.spline as sp
import global_defs
from data_utils.sketch_utils import get_subdirs, get_allfiles, sketch_std
import data_utils.sketch_vis as vis


class SketchDataset(Dataset):
    """
    定位文件的路径如下：
    root
    ├─ train
    │   ├─ Bushes
    │   │   ├─0.obj
    │   │   ├─1.obj
    │   │   ...
    │   │
    │   ├─ Clamps
    │   │   ├─0.obj
    │   │   ├─1.obj
    │   │   ...
    │   │
    │   ...
    │
    ├─ test
    │   ├─ Bushes
    │   │   ├─0.obj
    │   │   ├─1.obj
    │   │   ...
    │   │
    │   ├─ Clamps
    │   │   ├─0.obj
    │   │   ├─1.obj
    │   │   ...
    │   │
    │   ...
    │

    """
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\unified_sketch',
                 is_train=True,
                 data_argumentation=False
                 ):

        print('sketch dataset, from:' + root)
        self.data_augmentation = data_argumentation

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # 获取全部类别列表，即 inner_root 内的全部文件夹名
        category_all = get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        for item in category_path:  # item 为字典的键，即类型‘plane','car'
            for fn in category_path[item]:  # fn 为每类点云对应的文件路径
                self.datapath.append((item, fn))  # item：类型（‘plane','car'）

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # 用整形0,1,2,3等代表具体类型‘plane','car'等，此时字典category_path中的键值没有用到，self.classes的键为‘plane'或'car'，值为0,1
        print(self.classes)
        print('number of instance all:', len(self.datapath))

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]  # (‘plane’, Path1)
        cls = self.classes[fn[0]]  # 表示类别的整形数字

        # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
        sketch_data = np.loadtxt(fn[1], delimiter=',')

        # 2D coordinates
        coordinates = sketch_data[:, :2]

        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        # rotate and move
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            coordinates = coordinates @ rotation_matrix
            coordinates += np.random.normal(0, 0.02, size=coordinates.shape)

        return coordinates, cls

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)

    @staticmethod
    def check_format(dir_path, n_points_all):
        sci_float_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?,-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        format_fit = True

        files_all = get_allfiles(dir_path, 'txt')

        for c_file in tqdm(files_all, total=len(files_all)):
            data_read = np.loadtxt(c_file, delimiter=',')
            if data_read.shape[0] != n_points_all:
                print(f'file {c_file} not fir line count')
                format_fit = False
                break

            with open(c_file, 'r') as f:
                for c_line in f.readlines():
                    c_line = c_line.strip()

                    if not sci_float_pattern.match(c_line):
                        print(c_line, '不符合点云文件格式')
                        print('not fit line: ', c_line.strip())

                        format_fit = False
                        break

        return format_fit


class DiffDataset(Dataset):
    """
    diffusion 数据集加载
    读取 root 文件夹下的全部 txt 文件
    """
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\unified_sketch',
                 data_argumentation=False
                 ):

        print('sketch dataset, from:' + root)
        self.data_augmentation = data_argumentation
        self.datapath = get_allfiles(root)
        print('number of instance all:', len(self.datapath))

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]  # (‘plane’, Path1)

        # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
        sketch_data = np.loadtxt(fn, delimiter=',')

        # 2D coordinates
        coordinates = sketch_data[:, :2]

        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        # rotate and move
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            coordinates = coordinates @ rotation_matrix
            coordinates += np.random.normal(0, 0.02, size=coordinates.shape)

        return coordinates

    def __len__(self):
        return len(self.datapath)


class QuickdrawDataset(Dataset):
    def __init__(self, root=r'D:\document\DeepLearning\DataSet\quickdraw\sketchrnn_apple.npz', mode='train', max_seq_length=1000):
        """
        QuickDraw 数据集中 0表示抬笔，即该点为该笔划最后一个点，1表示落笔
        """
        data_all = np.load(str(root), encoding='latin1', allow_pickle=True)
        dataset = data_all[mode]

        data = []
        # We iterate through each of the sequences and filter
        for seq in dataset:
            # Filter if the length of the sequence of strokes is within our range
            if 10 < len(seq) <= max_seq_length:
                # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                # Convert to a floating point array and add to `data`
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        # We then calculate the scaling factor which is the
        # standard deviation of ($\Delta x$, $\Delta y$) combined.
        # Paper notes that the mean is not adjusted for simplicity,
        # since the mean is anyway close to $0$.
        scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        # Get the longest sequence length among all sequences
        longest_seq_len = max([len(seq) for seq in data])

        # We initialize PyTorch data array with two extra steps for start-of-sequence (sos)
        # and end-of-sequence (eos).
        # Each step is a vector $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # Only one of $p_1, p_2, p_3$ is $1$ and the others are $0$.
        # They represent *pen down*, *pen up* and *end-of-sequence* in that order.
        # $p_1$ is $1$ if the pen touches the paper in the next step.
        # $p_2$ is $1$ if the pen doesn't touch the paper in the next step.
        # $p_3$ is $1$ if it is the end of the drawing.
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)
        # The mask array needs only one extra-step since it is for the outputs of the
        # decoder, which takes in `data[:-1]` and predicts next step.
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
            # $p_1$
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            # $p_2$
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            # $p_3$
            self.data[i, len_seq + 1:, 4] = 1
            # Mask is on until end of sequence
            self.mask[i, :len_seq + 1] = 1

        # Start-of-sequence is $(0, 0, 1, 0, 0)$
        self.data[:, 0, 2] = 1

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample"""
        return self.data[idx], self.mask[idx]

    def get_sketch(self, idx: int, is_show_sketch=False):
        sketch, mask = self.__getitem__(idx)
        sketch = sketch[:, :3]

        # 去掉第一行最后一行
        sketch = sketch[1:-1, :]
        mask = mask[1:]

        sketch = sketch[mask == 1]
        sketch = sketch.numpy()

        xy_data = sketch[:, :2]
        xy_data = np.cumsum(xy_data, axis=0)
        xy_data = sketch_std(xy_data)

        sketch[:, :2] = xy_data

        # sketch = sketch_std(sketch)

        if is_show_sketch:
            vis.vis_sketch_data(sketch, global_defs.pen_up, global_defs.pen_down, is_scale=False)

        return sketch

    def save_std(self, target_root):
        for i in tqdm(range(len(self.data)), total=len(self.data)):
            sketch = self.get_sketch(i)
            np.savetxt(os.path.join(target_root, f'{i}.txt'), sketch, delimiter=',')


def std_unify(std_root: str, min_pnt: int = global_defs.n_stk * 2, is_mix_proc: bool = True):
    """
    将单个 std 草图转化为 unified_std 草图
    :param std_root: [n, 4]
    :param min_pnt: 处理后点数低于该数值的草图剔除
    :param is_mix_proc: 是否反复进行笔划拆分、合并、实现笔划的长度尽量相等
    :return:
    """

    def distance(p1, p2):
        """
        计算两点之间的欧氏距离
        """
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def search_nearest_stroke(stroke_list, end_point, given_idx):
        """
        找到stroke_list中stroke_idx对应的笔划中，end_point点最近的笔划，并返回是合并到起点还是终点
        :param stroke_list:
        :param end_point:
        :param given_idx:
        :return: closest_idx, closet_dist, is_connect_start
        """
        # 计算最短笔划与其他笔划的起始点距离，找到最近的笔划
        closest_idx_start = -1
        min_distance_start = float('inf')

        closest_idx_end = -1
        min_distance_end = float('inf')

        for i in range(len(stroke_list)):
            if i == given_idx:  # 跳过自身
                continue
            dist_start = distance(end_point, stroke_list[i][0])
            if dist_start <= min_distance_start:
                min_distance_start = dist_start
                closest_idx_start = i

            dist_end = distance(end_point, stroke_list[i][-1])
            if dist_end <= min_distance_end:
                min_distance_end = dist_end
                closest_idx_end = i

        if min_distance_start <= min_distance_end:
            if min_distance_start == float('inf'):
                raise ValueError('inf dist occurred')

                # print(stroke_list)
                # stroke_list = np.concatenate(stroke_list, axis=0)
                # np.savetxt('error_stk', stroke_list, delimiter=',')
                # exit(0)

            return closest_idx_start, min_distance_start, True
        else:
            if min_distance_start == float('inf'):
                raise ValueError('inf dist occurred 2')

                # print(stroke_list)
                # stroke_list = np.concatenate(stroke_list, axis=0)
                # np.savetxt('error_stk', stroke_list, delimiter=',')
                # exit(0)

            return closest_idx_end, min_distance_end, False

    def single_merge(stroke_list, dist_gap=0.2):
        """
        将草图中最短的一个笔划合并到其他笔划
        :param stroke_list:
        :param dist_gap: 若某个笔划距其它笔划的最近距离大于该值，不合并
        :return:
        """
        # stroke_list = copy.deepcopy(stk_list)

        # 因距离其它笔画太远不合并的笔划
        stk_cannot_merge = []
        ita_count = 0

        while True:
            if (len(stk_cannot_merge) + len(stroke_list)) > 2 and len(stroke_list) < 2:
                raise ValueError('所有笔划均距其它笔划过远，无法合并，请尝试增加笔划数')

            # 防止迭代次数过多陷入死循环
            if ita_count > 200:
                # print('到达最大迭代次数，不合并该笔划')
                # warnings.warn('cannot merge min stroke')
                raise ValueError('到达最大迭代次数，不合并该笔划')
            else:
                ita_count += 1

            # 找到点数最少的笔划索引
            min_idx = min(range(len(stroke_list)), key=lambda i: len(stroke_list[i]))
            min_stroke = stroke_list[min_idx]

            # if len(min_stroke) == 0:
            #     print(f'file root: {std_root}, 出现最短笔划点数为零')
            #     exit(0)

            min_start = min_stroke[0]  # 最短笔划的起始点
            min_end = min_stroke[-1]  # 最短笔划的起始点

            cidx_st, dist_st, is_ct_st_st = search_nearest_stroke(stroke_list, min_start, min_idx)
            cidx_ed, dist_ed, is_ct_st_ed = search_nearest_stroke(stroke_list, min_end, min_idx)

            # 如果最近的笔画距离其它笔画过远，不合并
            if min(dist_st, dist_ed) > dist_gap:
                stk_cannot_merge.append(min_stroke)
                del stroke_list[min_idx]

            else:
                if dist_st <= dist_ed:
                    closest_idx = cidx_st
                    is_this_start = True
                    is_target_start = is_ct_st_st
                else:
                    closest_idx = cidx_ed
                    is_this_start = False
                    is_target_start = is_ct_st_ed

                # 情形1：起点到起点，target不动，this调转后拼接在前面：
                target_stk = stroke_list[closest_idx]
                if is_this_start and is_target_start:
                    min_stroke = np.flip(min_stroke, axis=0)
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形2：起点到终点，this拼接在target后面：
                elif is_this_start and (not is_target_start):
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                # 情形3：终点到起点，this拼接在target前面：
                elif (not is_this_start) and is_target_start:
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形4：终点到终点，target不动，this调转后拼接在后面：
                elif (not is_this_start) and (not is_target_start):
                    min_stroke = np.flip(min_stroke, axis=0)
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                else:
                    warnings.warn('error occurred when stroke merged')
                    raise ValueError('error occurred when stroke merged')

                del stroke_list[min_idx]  # 删除已合并的笔划

                # 加入之前不能合并的距其它笔划过远的笔划
                if len(stk_cannot_merge) != 0:
                    stroke_list = stroke_list + stk_cannot_merge

                return stroke_list

    def single_split(stroke_list):
        """
        将草图中最长的笔画对半分割成两个
        :param stroke_list:
        :return:
        """
        # stroke_list = copy.deepcopy(stk_list)

        # Find the array with the maximum number of rows
        largest_idx = max(range(len(stroke_list)), key=lambda i: stroke_list[i].shape[0])
        largest_array = stroke_list[largest_idx]

        # Split the largest array into two halves
        split_point = largest_array.shape[0] // 2
        first_half = largest_array[:split_point, :]
        second_half = largest_array[split_point:, :]

        # Replace the largest array with the two halves
        del stroke_list[largest_idx]
        stroke_list.append(first_half)
        stroke_list.append(second_half)

        return stroke_list

    # 读取std草图数据
    sketch_data = np.loadtxt(std_root, delimiter=',')
    sketch_data[-1, 2] = global_defs.pen_down

    # 去掉点数过少的笔划
    sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # 去掉笔划上距离过近的点
    sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= min_pnt:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{std_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉过短的笔划
    strokes = sp.stroke_len_filter(strokes, 0.1)

    # 重采样，使得点之间的距离近似相等
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        median_ratio=0.1,
        approx_mode='uni-arclength'
    )

    # 先处理笔划数
    # 大于指定数值，将长度较短的笔划连接到最近的笔划
    n_stk = global_defs.n_stk

    if len(strokes) > n_stk:
        while len(strokes) > n_stk:
            strokes = single_merge(strokes)

    # 小于指定数值，拆分点数较多的笔划
    elif len(strokes) < n_stk:
        while len(strokes) < n_stk:
            strokes = single_split(strokes)

    if len(strokes) != global_defs.n_stk:
        raise ValueError(f'error stroke number: {len(strokes)}')

    if is_mix_proc:
        # 不断拆分、合并笔划，使各笔划点数尽量接近
        n_ita = 0
        while True:
            before_var = sp.stroke_length_var(strokes)

            strokes = single_merge(strokes)
            strokes = single_split(strokes)

            after_var = sp.stroke_length_var(strokes)

            if after_var == before_var or n_ita > 150:
                # print('number of iteration: ', n_ita)
                break

            n_ita += 1

    if len(strokes) != global_defs.n_stk:
        raise ValueError(f'error stroke number final: {len(strokes)}, file: {std_root}')

    # 处理点数
    n_stk_pnt = global_defs.n_stk_pnt
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        min_sample=n_stk_pnt,
        max_sample=n_stk_pnt,
        approx_mode='linear-interp'
    )

    return strokes


def std_unify_batched(source_dir=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt', target_dir=r'D:\document\DeepLearning\DataSet\unified_sketch', is_mix_proc=True):
    """
    将 source_dir 文件夹中的全部 std 草图转化为 unified_std 草图，并保存到 target_dir
    :param source_dir:
    :param target_dir:
    :return:
    """
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)

    files_all = get_allfiles(source_dir)

    for c_file in tqdm(files_all, total=len(files_all)):
        try:
            sketch_data = std_unify(c_file, is_mix_proc)
        except:
            warnings.warn(f'error occurred when trans {c_file}, has skipped!')
            continue

        # sketch_data = std_unify(c_file, is_mix_proc)
        if sketch_data is not None:

            sketch_data = np.concatenate(sketch_data, axis=0)
            sketch_data = sketch_data[:, :2]

            transed_npnts = len(sketch_data)
            if transed_npnts == global_defs.n_stk * global_defs.n_stk_pnt:

                target_save_path = c_file.replace(source_dir, target_dir)
                np.savetxt(target_save_path, sketch_data, delimiter=',')
            else:
                warnings.warn(f'current point number is {transed_npnts}, skip file trans: {c_file}')


def travese_quickdraw(root):
    """
    显示root对应的npz文件里的所有的草图
    :param root: quickdraw 数据集文件路径，例如 r'D:\quickdraw\sketchrnn_airplane.full.npz'
    :return:
    """
    data = QuickdrawDataset(root=root)
    instance_all = len(data)
    print('instance all: ', instance_all)

    for i in range(instance_all):
        data.get_sketch(i, is_show_sketch=True)


def quickdraw_to_std(quickdraw_root, std_root):
    """
    将quickdraw草图转化为std草图
    :param quickdraw_root: quickdraw 数据集文件路径，例如 r'D:\quickdraw\sketchrnn_airplane.full.npz'
    :param std_root: std 数据集文件夹路径，例如 r'D:\std
    :return: None
    """
    os.makedirs(std_root, exist_ok=True)

    print('clear folder: ', std_root)
    shutil.rmtree(std_root)

    os.makedirs(std_root, exist_ok=True)

    adataset = QuickdrawDataset(root=quickdraw_root)
    adataset.save_std(std_root)


if __name__ == '__main__':
    # travese_quickdraw(r'D:\document\DeepLearning\DataSet\quickdraw\sketchrnn_airplane.full.npz')

    # path = r'D:\document\DeepLearning\DataSet\quickdraw\sketchrnn_apple.npz'
    # # Load the numpy file
    # datasets = np.load(str(path), encoding='latin1', allow_pickle=True)
    #
    # print(datasets.keys())
    #
    # print(datasets['train'].shape)
    # print(datasets['test'].shape)
    # print(datasets['valid'].shape)

    # --------------------- quickdraw 转 txt
    # adataset = QuickdrawDataset(root=r'D:\document\DeepLearning\DataSet\quickdraw\sketchrnn_moon.full.npz')
    # adataset.save_std(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\moon')

    # --------------------- 草图标准化
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple', r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32')
    std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch\sketch_txt',
                      r'D:\document\DeepLearning\DataSet\unified_sketch')

    # quickdraw_to_std(r'D:\document\DeepLearning\DataSet\quickdraw\sketchrnn_apple.full.npz', r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple')




    # sks = std_unify(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\60686.txt')
    # vis.show_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\3514.txt')


    # # vis.show_sketch_list(sks)
    #
    # sks = np.concatenate(sks, axis=0)
    # sks = sks[:, :2]
    #
    # transed_npnts11 = len(sks)
    # if transed_npnts11 == global_defs.n_stk * global_defs.n_stk_pnt:
    #
    #     pass
    # else:
    #     warnings.warn(f'current point number is {transed_npnts11}, skip file trans')


    # sk_file = r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\10156.txt'
    # sketch_data = np.loadtxt(sk_file, delimiter=',')
    # sketch_data = std_unify(sketch_data, global_defs.n_stk, global_defs.n_stk_pnt)

    # sketch_std(np.loadtxt('error_stk'))

    pass



