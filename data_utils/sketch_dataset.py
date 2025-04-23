"""
草图数据集加载及转化相关

建议读取的 npz 文件中存储相对坐标，txt 和 svg 文件存储绝对坐标
因为 QuickDraw 数据集的 npz 文件存储相对坐标
TU_Berlin 数据集中的 svg 文件中存储绝对坐标

"""

import random
import math
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm
import shutil
from PIL import Image
from torchvision import transforms

import global_defs
from data_utils.sketch_utils import get_subdirs, get_allfiles, sketch_std
import data_utils.vis as vis
from encoders.PointBERT_ULIP2 import create_pretrained_pointbert
from data_utils import preprocess as pp
from data_utils import sketch_utils as du


class SketchDatasetCls(Dataset):
    """
    无需分割训练集、测试集，将数据放在一起即可自动按比例划分
    支持 svg 和 txt 和 png 格式的草图读取
    测试是使用 TU-Berlin 数据集，其中的 svg 为绝对坐标，如果你的 svg 文件存储的是相对坐标，请注意修改
    定位文件的路径如下：
    root
    ├─ Bushes
    │   ├─0.suffix
    │   ├─1.suffix
    │   ...
    │
    ├─ Clamps
    │   ├─0.suffix
    │   ├─1.suffix
    │   ...
    │
    ├─ Bearing
    │   ├─0.suffix
    │   ├─1.suffix
    │   ...
    │
    ...

    """
    def __init__(self,
                 root,
                 test_ratio=0.2,
                 suffix='svg',
                 is_random_divide=False,
                 data_mode='train',
                 back_mode='STK',
                 coor_mode='ABS',
                 n_max_len=200,
                 img_size=(224, 224)
                 ):
        """
        :param root:
        :param test_ratio: 测试集占总数据比例
        :param suffix: 数据文件后缀 ['svg', 'txt']
        :param is_random_divide: 分割训练集测试集时是否随机
        :param data_mode: ['train', 'test'], 区分训练集和测试集
        :param back_mode: ['STK', 'S5', 'IMG'].
            'STK': [n_stk, n_stk_pnt, 2],
            'S5': data: [n_max_len, 5], mask: [n_max_len, ]
            'IMG': [3, width, height]
        :param coor_mode: ['ABS', 'REL']
            'ABS': 绝对坐标
            'REL': 相对坐标
        :param n_max_len:
        :param img_size: 图片大小
        """
        print('sketch dataset total, from:' + root, '\n')
        assert data_mode == 'train' or data_mode == 'test'

        self.data_mode = data_mode
        self.back_mode = back_mode
        self.coor_mode = coor_mode
        self.n_max_len = n_max_len
        self.img_size = img_size
        self.suffix = suffix

        # 获取全部类别列表，即 root 内的全部文件夹名
        category_all = get_subdirs(root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(root, c_class)
            file_path_all = get_allfiles(class_root, suffix)

            category_path[c_class] = file_path_all

        self.datapath_train = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        self.datapath_test = []
        for item in category_path:  # item 为字典的键，即类型‘plane','car'
            c_class_path_list = category_path[item]

            if is_random_divide:
                random.shuffle(c_class_path_list)

            n_instance = len(c_class_path_list)
            n_test = math.ceil(test_ratio * n_instance)

            for i in range(n_instance):
                if i < n_test:
                    self.datapath_test.append((item, c_class_path_list[i]))
                else:
                    self.datapath_train.append((item, c_class_path_list[i]))

        # 用整形 0,1,2,3 等代表具体类型‘plane','car'等，
        # 此时字典 category_path 中的值没有用到，self.classes的键为 ‘plane' 或 'car' ，值为0, 1, 2, ...
        self.classes = dict(zip(sorted(category_path), range(len(category_path))))
        print(self.classes, '\n')

        print('number of training instance all:', len(self.datapath_train))
        print('number of testing instance all:', len(self.datapath_test))
        print('number of classes all:', len(self.classes), '\n')

    def __getitem__(self, index):
        """
        :return:
        STK: [n_stk, n_stk_pnt, 2], Null
        S5: [n_max_len, 5], [n_max_len, ]
        """

        while True:
            try:
                sketch_cube, mask, cls = self.get_data(index)
                return sketch_cube, mask, cls
            except:
                index = self.next_index(index)

    def get_data(self, index):
        if self.data_mode == 'train':
            datapath = self.datapath_train
        elif self.data_mode == 'test':
            datapath = self.datapath_test
        else:
            raise TypeError('error dataset mode')

        class_key, file_root = datapath[index]  # (‘plane’, Path1)
        cls = self.classes[class_key]  # 表示类别的整形数字

        if self.back_mode == 'STK':
            sketch_cube = pp.preprocess_force_seg_merge(file_root)
            mask = cls
        elif self.back_mode == 'S5':
            sketch_cube, mask = du.sketch_file_to_s5(file_root, self.n_max_len, self.coor_mode)

        elif self.back_mode == 'IMG':
            sketch_cube = du.img_read(file_root, self.img_size)
            mask = cls
        else:
            raise TypeError('error back mode')

        return sketch_cube, mask, cls

    def next_index(self, index):
        max_index = self.__len__()

        if index + 1 < max_index:
            return index + 1
        else:
            return 0

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.datapath_train)
        elif self.data_mode == 'test':
            return len(self.datapath_test)
        else:
            raise TypeError('error dataset mode')

    def n_classes(self):
        return len(self.classes)

    def train(self):
        self.data_mode = 'train'

    def eval(self):
        self.data_mode = 'test'


class QuickDrawDiff(Dataset):
    """
    读取 quickdraw 数据集，不区分类别，用于训练 diffusion
    单次读取一个 npz 文件，即一个类别
    """
    def __init__(self, root, back_mode='STK', coor_mode='ABS', max_len=200, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up):
        """
        将草图数据读取到该类的数组中
        :param root: npz 文件路径
        :param back_mode: ['STK', 'STD', 'S5']
            'STK': [n_stk, n_stk_pnt, 2]
            'STD': [n, 3] (x, y, s)
            'S5': data: [N, 5] (x, y, s1, s2, s3). mask: [N, ]. N = max_len + 2
        :param coor_mode:
        :param max_len:
        :param pen_down:
        :param pen_up:
        """
        print('QuickDrawDiff Dataset, from:', root)
        self.sketch_all = []
        self.mask_all = []

        if back_mode == 'S5':
            print('loading npz files')
            sketch_train, mask_train = du.npz_read(root, 'train', back_mode, coor_mode, max_len, pen_down, pen_up)
            sketch_test, mask_test = du.npz_read(root, 'test', back_mode, coor_mode, max_len, pen_down, pen_up)
            sketch_valid, mask_valid = du.npz_read(root, 'valid', back_mode, coor_mode, max_len, pen_down, pen_up)

            self.sketch_all.extend(sketch_train)
            self.sketch_all.extend(sketch_test)
            self.sketch_all.extend(sketch_valid)

            self.mask_all.extend(mask_train)
            self.mask_all.extend(mask_test)
            self.mask_all.extend(mask_valid)

        elif back_mode == 'STK' or back_mode == 'STD':
            print('loading npz files')
            sketch_train = du.npz_read(root, 'train', 'STD', coor_mode, max_len, pen_down, pen_up)[0]
            sketch_test = du.npz_read(root, 'test', 'STD', coor_mode, max_len, pen_down, pen_up)[0]
            sketch_valid = du.npz_read(root, 'valid', 'STD', coor_mode, max_len, pen_down, pen_up)[0]

            self.sketch_all.extend(sketch_train)
            self.sketch_all.extend(sketch_test)
            self.sketch_all.extend(sketch_valid)

            tmp_sketch_list = []
            if back_mode == 'STK':
                print('converting STD to STK')

                for c_sketch in tqdm(self.sketch_all):
                    tmp_sketch_list.append(pp.preprocess_force_seg_merge(c_sketch))

                self.sketch_all = tmp_sketch_list

        else:
            raise TypeError('error back mode')

        print('instance all: ', len(self.sketch_all))

    def __getitem__(self, index):
        sketch = self.sketch_all[index]
        mask = self.mask_all[index] if len(self.mask_all) > 0 else 0

        return sketch, mask

    def __len__(self):
        return len(self.sketch_all)


class QuickDrawCls(Dataset):
    """
    读取 quickdraw 数据集，用于分类
    npz文件单个文件包含了一个类中所有数据，且训练集测试集已划分好
    原本包含 'train', 'test', 'valid' 三个分集
    这里将 'train' 和 'valid' 合并为训练集，'test' 为测试集

    文件夹结构组织如下：
    root
    ├─ Bushes.npz
    ├─ Clamps.npz
    ├─ Bearing.npz
    ├─ Gear.npz
    ├─ ...
    │
    ...

    """
    def __init__(self,
                 root_npz,
                 data_mode='train',
                 back_mode='STK',
                 coor_mode='ABS',
                 max_len=200,
                 is_process_in_init=False,
                 pen_down=global_defs.pen_down,
                 pen_up=global_defs.pen_up
                 ):
        """
        将草图数据读取到该类的数组中
        :param root_npz:
        :param data_mode: ['train', 'test']
        :param back_mode: ['STK', 'STD', 'S5']
            'STK': [n_stk, n_stk_pnt, 2]
            'S5': data: [N, 5] (x, y, s1, s2, s3). mask: [N, ]. N = max_len + 2
        :param coor_mode:
        :param max_len:
        :param is_process_in_init: 是否在 __init__ 方法中将 'STD' 转化为 'STK'
            False: 可提升加载速度，但训练时进行转化，降低训练速度，整体看降低训练速度
            True: 需要极长加载速度，训练时无需转化，整体看提升训练速度
        :param pen_down:
        :param pen_up:
        """
        print('QuickDrawCls Dataset, from:', root_npz)
        assert data_mode == 'train' or data_mode == 'test'

        self.data_mode = data_mode
        self.back_mode = back_mode
        self.is_process_in_init = is_process_in_init
        self.data_train = []
        self.data_test = []
        category_all = []
        npz_all = get_allfiles(root_npz, 'npz')

        print('loading npz file ...')
        for c_pnz in tqdm(npz_all):

            c_class = os.path.basename(c_pnz).split('.')[0]
            category_all.append(c_class)

            if back_mode == 'S5':
                back_mode_alt = 'S5'
            elif back_mode == 'STK':
                back_mode_alt = 'STD'
            else:
                raise TypeError('error back mode')

            sk_train, mk_train = du.npz_read(c_pnz, 'train', back_mode_alt, coor_mode, max_len, pen_down, pen_up)
            sk_test, mk_test = du.npz_read(c_pnz, 'test', back_mode_alt, coor_mode, max_len, pen_down, pen_up)
            sk_valid, mk_valid = du.npz_read(c_pnz, 'valid', back_mode_alt, coor_mode, max_len, pen_down, pen_up)

            sk_train.extend(sk_valid)
            mk_train.extend(mk_valid)

            if back_mode == 'S5':
                c_train = [(c_class, sk, mk) for sk, mk in zip(sk_train, mk_train)]
                c_test = [(c_class, sk, mk) for sk, mk in zip(sk_test, mk_test)]

            elif back_mode == 'STK':
                if self.is_process_in_init:
                    c_train = []
                    c_test = []

                    for c_instance in sk_train:
                        c_skh_stk = pp.preprocess_force_seg_merge(c_instance)
                        c_train.append((c_class, c_skh_stk, 0))

                    for c_instance in sk_test:
                        c_skh_stk = pp.preprocess_force_seg_merge(c_instance)
                        c_test.append((c_class, c_skh_stk, 0))

                else:
                    c_train = [(c_class, sk, 0) for sk in sk_train]
                    c_test = [(c_class, sk, 0) for sk in sk_test]

            else:
                raise TypeError('error back mode')

            self.data_train.extend(c_train)
            self.data_test.extend(c_test)

        self.classes = dict(zip(sorted(category_all), range(len(category_all))))
        print(self.classes, '\n')

        print('number of training instance all:', len(self.data_train))
        print('number of testing instance all:', len(self.data_test))

    def __getitem__(self, index):
        if self.data_mode == 'train':
            datapath = self.data_train
        elif self.data_mode == 'test':
            datapath = self.data_test
        else:
            raise TypeError('error dataset mode')

        cls, data, mask = datapath[index]
        cls = self.classes[cls]

        # 需要将 'STD' 转化为 'STK'
        if self.back_mode == 'STK' and not self.is_process_in_init:
            data = pp.preprocess_force_seg_merge(data)

        return data, mask, cls

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.data_train)
        elif self.data_mode == 'test':
            return len(self.data_test)
        else:
            raise TypeError('error dataset mode')

    def train(self):
        self.data_mode = 'train'

    def eval(self):
        self.data_mode = 'test'


class DiffDataset(Dataset):
    """
    diffusion 数据集加载
    读取 root 文件夹下的全部 txt 或者 svg 文件
    文件需记录绝对坐标
    """
    def __init__(self,
                 root,
                 suffix='txt',
                 back_mode='STK',
                 coor_mode='ABS',
                 n_max_len=200
                 ):
        """
        :param root:
        :param suffix:
        :param back_mode: ['STK', 'S5']
        :param coor_mode: ['ABS', 'REL']
        :param n_max_len:
        """
        print(f'diffusion dataset, from: {root}')

        self.back_mode = back_mode
        self.coor_mode = coor_mode
        self.n_max_len = n_max_len

        self.datapath = get_allfiles(root, suffix)

        print(f'number of instance all: {len(self.datapath)}')

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]

        if self.back_mode == 'STK':
            sketch_cube = pp.preprocess_force_seg_merge(fn)
            mask = 0
        elif self.back_mode == 'S5':
            sketch_cube, mask = du.sketch_file_to_s5(fn, self.n_max_len, self.coor_mode)
        else:
            raise TypeError('error back mode')

        return sketch_cube, mask

    def __len__(self):
        return len(self.datapath)


class RetrievalDataset(Dataset):
    def __init__(self, root, mode='train', max_seq_length=256, image_size=(224, 224), return_mode='S5'):
        """

        :param root:
        :param mode: ['train', 'test']
        :param max_seq_length:
        :param return_mode:
        """
        self.max_seq_length = max_seq_length
        self.return_mode = return_mode
        self.image_size = image_size

        png_root = os.path.join(root, 'sketch_png', mode)

        # 获取全部样本
        self.pngs_all = get_allfiles(png_root, 'png')

    def __getitem__(self, index):
        png_path = self.pngs_all[index]

        txt_path = png_path.replace('sketch_png', 'sketch_txt')
        txt_path = txt_path.replace('png', 'txt')

        image = Image.open(png_path).convert("RGB")  # 确保是 RGB 模式
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()  # 转换为 [C, H, W] 格式的 Tensor，值在 [0, 1] 之间
        ])
        tensor_image = transform(image)

        if self.return_mode == 'S5':
            txt_tensor, mask = du.sketch_file_to_s5(txt_path, self.max_seq_length)

        elif self.return_mode == 'STK':
            sketch_data = pp.preprocess_just_pad(txt_path)
            txt_tensor, mask = du.stroke_list_to_sketch_cube(sketch_data)

        else:
            raise ValueError('error return type')

        return tensor_image, txt_tensor, mask

    def __len__(self):
        return len(self.pngs_all)


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
        sketch = sketch[1: -1, :]
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


def sig_process(data_set, point_bert):
    loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=False, num_workers=4)

    for batch_id, data in tqdm(enumerate(loader, 0), total=len(loader)):
        xy, target, idx = data[0].float().cuda(), data[1].long().cuda(), data[-1].long().cuda()

        bs, n_skh_pnt, pnt_channel = xy.size()
        assert n_skh_pnt == global_defs.n_stk * global_defs.n_stk_pnt and pnt_channel == 2

        xy = xy.view(bs, global_defs.n_stk, global_defs.n_stk_pnt, 2)
        z = -0.5 * (xy[:, :, :, 0] + xy[:, :, :, 1]).unsqueeze(3)
        xyz = torch.cat([xy, z], dim=3)  # [bs, n_stk, n_stk_pnt, 3]
        xyz = xyz.view(bs * global_defs.n_stk, global_defs.n_stk_pnt, 3)  # [bs * n_stk, n_stk_pnt, 3]

        # zeros = torch.zeros(bs * global_defs.n_stk, global_defs.n_stk_pnt, 1, device=xy.device, dtype=xy.dtype)
        # xy = torch.cat([xy, zeros], dim=2)

        xyz = point_bert(xyz)  # [bs * n_stk, 512]
        xyz = xyz.view(bs, global_defs.n_stk, point_bert.channel_out)

        for i in range(bs):
            c_stk_coor = xyz[i, :, :]
            c_data_idx = idx[i].item()

            # 很具target找到对应的文件夹
            c_data_root = data_set.datapath[c_data_idx][1]

            c_stk_root = c_data_root.replace('.txt', '.stkcoor')

            np.savetxt(c_stk_root, c_stk_coor.cpu().numpy(), delimiter=',')


def add_stk_coor(dir_path):
    """
    为unified_std_sketch添加笔划特征，使用point_bert_ulip2
    :param dir_path: 分类数据集根目录
    :param bs: 分类数据集根目录
    :return:
    """

    train_dataset = SketchDataset(root=dir_path, is_train=True, is_back_idx=True)
    test_dataset = SketchDataset(root=dir_path, is_train=False, is_back_idx=True)

    point_bert = create_pretrained_pointbert(r'E:\document\DeepLearning\SDGraph\model_trained\pointbert_ulip2.pth').cuda()

    sig_process(train_dataset, point_bert)
    sig_process(test_dataset, point_bert)













if __name__ == '__main__':
    # adataset = DiffDataset(f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', shuffle_stk=True)
    #
    # for _ in range(10):
    #     adataset.vis_sketch(0)

    # quickdraw_to_std_batched(r'D:\document\DeepLearning\DataSet\quickdraw\raw', r'D:\document\DeepLearning\DataSet\quickdraw\txt')

    # add_stk_coor(r'D:\document\DeepLearning\DataSet\sketch_cad\unified_sketch_cad_stk32_stkpnt32')


    # adtast = SketchDataset2(root=data_root, is_train=True)



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



