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
import torch
import os
from tqdm import tqdm
import re
import shutil

import global_defs
from data_utils.sketch_utils import get_subdirs, get_allfiles, sketch_std
import data_utils.sketch_vis as vis
from encoders.PointBERT_ULIP2 import create_pretrained_pointbert


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
                 data_argumentation=False,
                 is_back_idx=False
                 ):

        print('sketch dataset, from:' + root)
        self.data_augmentation = data_argumentation
        self.is_back_idx = is_back_idx

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
        print('number of classes all:', len(self.classes))

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

        if self.is_back_idx:
            return coordinates, cls, index
        else:
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
                 shuffle_stk=False,  # 是否随机变换笔划顺序
                 data_aug=False  # 是否进行数据增强
                 ):

        self.data_aug = data_aug
        self.shuffle_stk = shuffle_stk
        self.datapath = get_allfiles(root)

        print(f'diffusion dataset, from: {root}')
        print(f'shuffle stroke: {self.shuffle_stk}, data argumentation: {self.data_aug}')
        print(f'number of instance all: {len(self.datapath)}')

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]
        sketch_data = np.loadtxt(fn, delimiter=',')

        # 2D coordinates
        coordinates = sketch_data[:, :2]

        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        # rotate and move
        if self.data_aug:
            theta = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            coordinates = coordinates @ rotation_matrix
            coordinates += np.random.normal(0, 0.02, size=coordinates.shape)

        # 随机变换笔划顺序
        if self.shuffle_stk:
            coordinates = coordinates.reshape([global_defs.n_stk, global_defs.n_stk_pnt, 2])
            np.random.shuffle(coordinates)
            coordinates = coordinates.reshape([global_defs.n_skh_pnt, 2])

        return coordinates

    def __len__(self):
        return len(self.datapath)

    def vis_sketch(self, idx):
        c_sketch = self.__getitem__(idx)
        vis.vis_unified_sketch_data(c_sketch)


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

        if is_show_sketch:
            vis.vis_sketch_data(sketch, global_defs.pen_up, global_defs.pen_down, is_scale=False)

        return sketch

    def save_std(self, target_root):
        for i in tqdm(range(len(self.data)), total=len(self.data)):
            sketch = self.get_sketch(i)
            np.savetxt(os.path.join(target_root, f'{i}.txt'), sketch, delimiter=',')


def travese_quickdraw(root_npz):
    """
    显示root对应的npz文件里的所有的草图
    :param root_npz: quickdraw 数据集文件路径，例如 r'D:\quickdraw\sketchrnn_airplane.full.npz'
    :return:
    """
    data = QuickdrawDataset(root=root_npz)
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


def quickdraw_to_std_batched(root_npz, root_txt):
    """
    将root_npz下的全部npz文件转化为std草图（非unified std草图）
    一个npz转对应一个文件夹
    :param root_npz:
    :param root_txt:
    :return:
    """

    # 找到全部文件
    npz_all = get_allfiles(root_npz, 'npz')

    for c_npz in npz_all:
        c_class = os.path.basename(c_npz).split('.')[0]
        print(c_class)
        c_class_dir = os.path.join(root_txt, c_class)
        os.makedirs(c_class_dir)

        quickdraw_to_std(c_npz, c_class_dir)


def add_stk_coor(dir_path):
    """
    为unified_std_sketch添加笔划特征，使用point_bert_ulip2
    :param dir_path: 分类数据集根目录
    :param bs: 分类数据集根目录
    :return:
    """
    def sig_process(data_set):
        loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=False, num_workers=4)

        for batch_id, data in tqdm(enumerate(loader, 0), total=len(loader)):
            xy, target, idx = data[0].float().cuda(), data[1].long().cuda(), data[2].long().cuda()

            bs, n_skh_pnt, pnt_channel = xy.size()
            assert n_skh_pnt == global_defs.n_stk * global_defs.n_stk_pnt and pnt_channel == 2

            xy = xy.view(bs, global_defs.n_stk, global_defs.n_stk_pnt, 2)
            xy = xy.view(bs * global_defs.n_stk, global_defs.n_stk_pnt, 2)
            zeros = torch.zeros(bs * global_defs.n_stk, global_defs.n_stk_pnt, 1, device=xy.device, dtype=xy.dtype)
            xy = torch.cat([xy, zeros], dim=2)

            xy = point_bert(xy)
            xy = xy.view(bs, global_defs.n_stk, point_bert.channel_out)

            for i in range(bs):
                c_stk_coor = xy[i, :, :]
                c_data_idx = idx[i].item()

                # 很具target找到对应的文件夹
                c_data_root = train_dataset.datapath[c_data_idx][1]

                c_stk_root = c_data_root.replace('.txt', '.stkcoor')

                np.savetxt(c_stk_root, c_stk_coor.cpu().numpy(), delimiter=',')

    train_dataset = SketchDataset(root=dir_path, is_train=True, is_back_idx=True)
    test_dataset = SketchDataset(root=dir_path, is_train=False, is_back_idx=True)

    point_bert = create_pretrained_pointbert(r'E:\document\DeepLearning\SDGraph\encoders\pointbert_ulip2.pth').cuda()

    sig_process(train_dataset)
    sig_process(test_dataset)













if __name__ == '__main__':
    # adataset = DiffDataset(f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', shuffle_stk=True)
    #
    # for _ in range(10):
    #     adataset.vis_sketch(0)

    # quickdraw_to_std_batched(r'D:\document\DeepLearning\DataSet\quickdraw\raw', r'D:\document\DeepLearning\DataSet\quickdraw\txt')

    add_stk_coor(r'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk32_stkpnt32')




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



