"""
草图数据集加载及转化相关

建议读取的 npz 文件中存储相对坐标，txt 和 svg 文件存储绝对坐标
因为 QuickDraw 数据集的 npz 文件存储相对坐标
TU_Berlin 数据集中的 svg 文件中存储绝对坐标

定义：
S5 草图格式: [n, 5] 每行为：(x, y, s1, s2, s3)
STD 草图格式: [n, 3] 每行为：(x, y, s)
STK 草图格式: [n_stk, n_stk_pnt, 2], n_stk: 草图笔划数，n_stk_pnt: 每个笔划点数

ABS 坐标格式: 绝对坐标
REL 坐标格式: 相对坐标，除第一个点外，其它坐标均为相对前一个点的坐标偏移量

"""

import random
import math
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool
from functools import partial
import numpy as np
from data_utils.vis import vis_sketch_orig, vis_s5_data

import global_defs
# from data_utils.preprocess import preprocess_force_seg_merge as prep
# from data_utils.preprocess import preprocess_split_merge_until as prep
# from data_utils.preprocess import resample_stake as prep
from data_utils.preprocess import preprocess_orig as prep
from data_utils import sketch_utils as du
from data_utils import sketch_file_read as fr
from data_utils import data_convert as dc


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
                 is_random_divide=False,
                 data_mode='train',
                 back_mode='STK',
                 coor_mode='ABS',
                 max_len=200,
                 img_size=(224, 224),
                 workers=8,
                 is_retrieval=False,
                 is_already_divided=False,
                 is_preprocess=True,
                 is_shuffle_stroke=False
                 ):
        """
        :param root:
        :param test_ratio: 测试集占总数据比例
        :param is_random_divide: 分割训练集测试集时是否随机
        :param data_mode: ['train', 'test'], 区分训练集和测试集
        :param back_mode: ['STK', 'S5', 'IMG'].
            'STK': [n_stk, n_stk_pnt, 2],
            'S5': data: [n_max_len, 5], mask: [n_max_len, ]
            'IMG': [3, width, height]
        :param coor_mode: ['ABS', 'REL']
            'ABS': 绝对坐标
            'REL': 相对坐标
        :param max_len:
        :param img_size: 图片大小
        :param workers:
        :param is_retrieval: 是否是检索
        :param is_already_divided: 训练集与测试集是否已经划分好
        :param is_preprocess: 是否需要进行预处理，即是否需要将 STD 草图转化为 STK，如果无需处理，说明文件里已转化好
        :param is_shuffle_stroke: 是否打乱笔划
        :return:
        """
        print('sketch dataset total, from:' + root + f'. using workers: {workers}')
        assert data_mode == 'train' or data_mode == 'test'

        self.data_mode = data_mode
        self.back_mode = back_mode
        self.coor_mode = coor_mode
        self.max_len = max_len
        self.img_size = img_size
        self.is_retrieval = is_retrieval
        self.is_preprocess = is_preprocess
        self.is_shuffle_stroke = is_shuffle_stroke

        # 训练测试集按给定比例划分
        train_dir = os.path.join(root, 'train')
        test_dir = os.path.join(root, 'test')

        if is_already_divided:
            print('train and test set is already divided')
            assert os.path.isdir(train_dir) and os.path.isdir(test_dir)

            datapath_train = self.get_file_path(train_dir)
            datapath_test = self.get_file_path(test_dir)
            category_path = du.get_subdirs(train_dir)

        # 训练测试集已划分好
        else:
            print('train and test set is not divided, auto divide train and test set')
            assert not os.path.isdir(train_dir) and not os.path.isdir(test_dir)
            datapath_train, datapath_test = self.get_file_path(root, True, is_random_divide, test_ratio)
            category_path = du.get_subdirs(root)

        if back_mode == 'STK' and not self.is_preprocess:
            self.data_train = datapath_train
            self.data_test = datapath_test

        else:
            # 数据预处理，防止后续重复处理
            worker_func = partial(self.load_data,
                                  back_mode=back_mode,
                                  coor_mode=coor_mode,
                                  max_len=max_len,
                                  img_size=img_size,
                                  is_retrieval=is_retrieval,
                                  is_shuffle_stroke=is_shuffle_stroke
                                  )

            if workers >= 2:
                with Pool(processes=workers) as pool:
                    self.data_train = list(tqdm(
                        pool.imap(worker_func, datapath_train),
                        total=len(datapath_train),
                        desc='processing training files')
                    )

                with Pool(processes=workers) as pool:
                    self.data_test = list(tqdm(
                        pool.imap(worker_func, datapath_test),
                        total=len(datapath_test),
                        desc='processing testing files')
                    )
            else:
                self.data_train = []
                for c_datapath_train in tqdm(datapath_train, total=len(datapath_train),
                                             desc='processing training files'):
                    self.data_train.append(worker_func(c_datapath_train))

                self.data_test = []
                for c_datapath_test in tqdm(datapath_test, total=len(datapath_test), desc='processing testing files'):
                    self.data_test.append(worker_func(c_datapath_test))

            # 删除异常值
            print('删除异常值')
            self.data_train = list(filter(lambda x: x is not None, self.data_train))
            self.data_test = list(filter(lambda x: x is not None, self.data_test))

        # 用整形 0,1,2,3 等代表具体类型‘plane','car'等，
        # 此时字典 category_path 中的值没有用到，self.classes的键为 ‘plane' 或 'car' ，值为 0, 1, 2, ...
        self.classes = dict(zip(sorted(category_path), range(len(category_path))))
        print(self.classes, '\n')

        print('number of training instance all:', len(self.data_train))
        print('number of testing instance all:', len(self.data_test))
        print('number of classes all:', len(self.classes), '\n')

    @staticmethod
    def get_file_path(folder_root, is_divide=False, is_random_divide=False, test_ratio=0.0):
        """
        在如下层级的目录中获取其中的文件，返回一个数组
        file_root
        ├─ Plane
        │   ├─0.suffix
        │   ├─1.suffix
        │   ...
        │
        ├─ Car
        │   ├─0.suffix
        │   ├─1.suffix
        │   ...
        │
        ├─ Chair
        │   ├─0.suffix
        │   ├─1.suffix
        │   ...
        │
        ...

        [('plane', root1), ('car', root2), ...] 即标签和路径的二联体
        :param folder_root:
        :param is_divide:
        :param is_random_divide: 在分割训练集测试集时是否随机
        :param test_ratio:
        :return:
        """
        if is_divide:
            category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

            # 获取全部类别列表，即 root 内的全部文件夹名
            category_all = du.get_subdirs(folder_root)

            for c_class in category_all:
                class_root = os.path.join(folder_root, c_class)
                file_path_all = du.get_allfiles(class_root, suffix=None)

                category_path[c_class] = file_path_all

            datapath_train = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
            datapath_test = []

            for item in category_path:  # item 为字典的键，即类型‘plane','car'
                c_class_path_list = category_path[item]

                if is_random_divide:
                    random.shuffle(c_class_path_list)

                n_instance = len(c_class_path_list)
                n_test = math.ceil(test_ratio * n_instance)

                for i in range(n_instance):
                    if i < n_test:
                        datapath_test.append((item, c_class_path_list[i]))
                    else:
                        datapath_train.append((item, c_class_path_list[i]))

            return datapath_train, datapath_test

        else:
            # 获取全部类别列表，即 file_root 内的全部文件夹名
            category_all = du.get_subdirs(folder_root)
            category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

            for c_class in category_all:
                class_root = os.path.join(folder_root, c_class)
                file_path_all = du.get_allfiles(class_root, suffix=None)

                category_path[c_class] = file_path_all

            datapath = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
            for item in category_path:  # item 为字典的键，即类型‘plane','car'
                for fn in category_path[item]:  # fn 为每类点云对应的文件路径
                    datapath.append((item, fn))  # item：类型（‘plane','car'）

            return datapath

    @staticmethod
    def load_data(cls_path, back_mode, coor_mode, max_len, img_size, is_retrieval, is_shuffle_stroke):
        """
        从文件里读取数据，并进行预处理
        支持三种类型数据读取
        :param cls_path: ('class', file_path)
        :param back_mode:
        :param coor_mode:
        :param max_len:
        :param img_size:
        :param is_retrieval:
        :param is_shuffle_stroke:
        :return:
        """
        class_name, file_root = cls_path

        try:
            if back_mode == 'STK':
                sketch_cube = prep(file_root, is_shuffle_stroke=is_shuffle_stroke)
                if is_retrieval:
                    sketch_img = dc.std_to_tensor_img(file_root)
                    return sketch_img, sketch_cube, 0

                else:
                    return class_name, sketch_cube, 0

            elif back_mode == 'S5':
                sketch_cube, mask = dc.sketch_file_to_s5(file_root, max_len, coor_mode, is_shuffle_stroke)
                if is_retrieval:
                    sketch_img = dc.std_to_tensor_img(file_root)
                    return sketch_img, sketch_cube, mask

                else:
                    return class_name, sketch_cube, mask

            elif back_mode == 'IMG':
                sketch_cube = fr.img_read(file_root, img_size)
                return class_name, sketch_cube, 0
            else:
                raise TypeError('error back mode')
        except:
            return None

    def __getitem__(self, index):
        """
        :return:
        STK: [n_stk, n_stk_pnt, 2], Null
        S5: [n_max_len, 5], [n_max_len, ]
        """
        if self.data_mode == 'train':
            data = self.data_train
        elif self.data_mode == 'test':
            data = self.data_test
        else:
            raise TypeError('error dataset mode')

        if not self.is_preprocess:
            class_key, data_path = data[index]
            sketch_cube = np.loadtxt(data_path, delimiter=',')
            sketch_cube = sketch_cube.reshape(global_defs.n_stk, global_defs.n_stk_pnt, 2)

            if self.is_shuffle_stroke:
                np.random.shuffle(sketch_cube)

            mask = 0

        else:
            class_key, sketch_cube, mask = data[index]

        if self.is_retrieval:
            return class_key, sketch_cube, mask

        else:
            cls = self.classes[class_key]  # 表示类别的整形数字
            return sketch_cube, mask, cls

    def next_index(self, index):
        max_index = self.__len__()

        if index + 1 < max_index:
            return index + 1
        else:
            return 0

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.data_train)
        elif self.data_mode == 'test':
            return len(self.data_test)
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
    def __init__(self,
                 root,
                 back_mode='STK',
                 coor_mode='ABS',
                 max_len=200,
                 workers=4,
                 pen_down=global_defs.pen_down,
                 pen_up=global_defs.pen_up):
        """
        将草图数据读取到该类的数组中
        :param root: npz 文件路径
        :param back_mode: ['STK', 'STD', 'S5']
            'STK': [n_stk, n_stk_pnt, 2]
            'STD': [n, 3] (x, y, s)
            'S5': data: [N, 5] (x, y, s1, s2, s3). mask: [N, ]. N = max_len + 2
        :param coor_mode:
        :param max_len:
        :param workers: 'STK' -> 'STD' 过程使用的线程数
        :param pen_down:
        :param pen_up:
        """
        print('QuickDrawDiff Dataset, from:', root, f'. using workers: {workers}')
        self.sketch_all = []
        self.mask_all = []

        print('loading npz files ...')
        if back_mode == 'S5':
            sketch_train, mask_train = fr.npz_read(root, 'train', back_mode, coor_mode, max_len, pen_down, pen_up)
            sketch_test, mask_test = fr.npz_read(root, 'test', back_mode, coor_mode, max_len, pen_down, pen_up)
            sketch_valid, mask_valid = fr.npz_read(root, 'valid', back_mode, coor_mode, max_len, pen_down, pen_up)

            self.sketch_all.extend(sketch_train)
            self.sketch_all.extend(sketch_test)
            self.sketch_all.extend(sketch_valid)

            self.mask_all.extend(mask_train)
            self.mask_all.extend(mask_test)
            self.mask_all.extend(mask_valid)

        elif back_mode == 'STK' or back_mode == 'STD':
            sketch_train = fr.npz_read(root, 'train', 'STD', coor_mode, max_len, pen_down, pen_up)[0]
            sketch_test = fr.npz_read(root, 'test', 'STD', coor_mode, max_len, pen_down, pen_up)[0]
            sketch_valid = fr.npz_read(root, 'valid', 'STD', coor_mode, max_len, pen_down, pen_up)[0]

            self.sketch_all.extend(sketch_train)
            self.sketch_all.extend(sketch_test)
            self.sketch_all.extend(sketch_valid)

            tmp_sketch_list = []
            if back_mode == 'STK':
                # 使用多进程处理数据
                if workers >= 2:
                    with Pool(processes=workers) as pool:
                        tmp_sketch_list = list(
                            tqdm(
                                pool.imap(prep, self.sketch_all),
                                total=len(self.sketch_all),
                                desc='converting STD to STK'
                            )
                        )
                else:
                    for c_sketch in tqdm(self.sketch_all, total=len(self.sketch_all), desc='converting STD to STK'):
                        tmp_sketch_list.append(prep(c_sketch))

                self.sketch_all = tmp_sketch_list

        else:
            raise TypeError('error back mode')

        # 删除异常值
        print('删除异常值')
        self.sketch_all = list(filter(lambda x: x is not None, self.sketch_all))

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
                 workers=4,
                 select=(1000, 100, 100),
                 is_random_select=False,
                 is_process_in_init=True,
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
        :param workers: 'STK' -> 'STD' 过程使用的线程数
        :param select: 从每类中选择的样本数。 = None: 不选取
            select[0]: 从 train
            select[1]: 从 valid
            select[2]: 从 test
        :param is_random_select: 选取样本时是否随机
        :param is_process_in_init: 是否在 __init__ 方法中将 'STD' 转化为 'STK'
            False: 可提升加载速度，但训练时进行转化，降低训练速度，整体看降低训练速度
            True: 需要极长加载速度，训练时无需转化，整体看提升训练速度
        :param pen_down:
        :param pen_up:

        """
        print('QuickDrawCls Dataset, from:' + root_npz + f'. using workers: {workers}')
        assert data_mode == 'train' or data_mode == 'test'

        self.data_mode = data_mode
        self.back_mode = back_mode
        self.is_process_in_init = is_process_in_init

        self.data_train = []
        self.data_test = []
        category_all = []

        # 固定其它参数
        worker_func = partial(self.load_npz,
                              back_mode=back_mode,
                              coor_mode=coor_mode,
                              max_len=max_len,
                              pen_down=pen_down,
                              pen_up=pen_up,
                              select=select,
                              is_random_select=is_random_select
                              )

        npz_all = du.get_allfiles(root_npz, 'npz')
        with Pool(processes=workers) as pool:
            # -> [(c_train, c_test)]
            results = list(tqdm(
                pool.imap(worker_func, npz_all),
                total=len(npz_all),
                desc='loading npz files')
            )

        for c_npz in npz_all:
            c_class = os.path.basename(c_npz).split('.')[0]
            category_all.append(c_class)

        for c_res in results:
            if c_res is not None:
                self.data_train.extend(c_res[0])
                self.data_test.extend(c_res[1])

        # 是否在init函数里进行转化
        if self.back_mode == 'STK' and self.is_process_in_init:
            # 使用多进程处理数据
            with Pool(processes=workers) as pool:
                self.data_train = list(
                    tqdm(
                        pool.imap(self.std_to_stk, self.data_train),
                        total=len(self.data_train),
                        desc='converting training set'
                    )
                )

            with Pool(processes=workers) as pool:
                self.data_test = list(
                    tqdm(
                        pool.imap(self.std_to_stk, self.data_test),
                        total=len(self.data_test),
                        desc='converting testing set'
                    )
                )

        # 删除异常值
        print('删除异常值')
        self.data_train = list(filter(lambda x: x is not None, self.data_train))
        self.data_test = list(filter(lambda x: x is not None, self.data_test))

        self.classes = dict(zip(sorted(category_all), range(len(category_all))))
        print('class_name, index map: ', self.classes, '\n')

        print('number of training instance all:', len(self.data_train))
        print('number of testing instance all:', len(self.data_test))

    @staticmethod
    def load_npz(c_npz, back_mode, coor_mode, max_len, pen_down, pen_up, select, is_random_select):
        c_class = os.path.basename(c_npz).split('.')[0]

        if back_mode == 'S5':
            back_mode_alt = 'S5'
        elif back_mode == 'STK':
            back_mode_alt = 'STD'
        else:
            raise TypeError('error back mode')

        sk_train, mk_train = fr.npz_read(c_npz, 'train', back_mode_alt, coor_mode, max_len, pen_down, pen_up)
        sk_test, mk_test = fr.npz_read(c_npz, 'test', back_mode_alt, coor_mode, max_len, pen_down, pen_up)

        # sk_valid, mk_valid = du.npz_read(c_npz, 'valid', back_mode_alt, coor_mode, max_len, pen_down, pen_up)
        # sk_train.extend(sk_valid)
        # mk_train.extend(mk_valid)

        if back_mode == 'S5':
            c_train = [(c_class, sk, mk) for sk, mk in zip(sk_train, mk_train)]
            c_test = [(c_class, sk, mk) for sk, mk in zip(sk_test, mk_test)]

            if select is not None:
                if is_random_select:
                    random.shuffle(c_train)
                    random.shuffle(c_test)

                c_train = c_train[:select[0]]
                c_test = c_test[:select[2]]

        elif back_mode == 'STK' or back_mode == 'STD':
            if select is not None:
                sk_train = sk_train[:select[0]]
                sk_test = sk_test[:select[2]]

            c_train = [(c_class, sk, 0) for sk in sk_train]
            c_test = [(c_class, sk, 0) for sk in sk_test]

        else:
            raise TypeError('error back mode')

        return c_train, c_test

    @staticmethod
    def std_to_stk(data_tup):
        """
        将单个 STD 草图转化成 STK 草图
        :param data_tup:
        :return:
        """
        try:
            res = prep(data_tup[1])
            return data_tup[0], res, 0
        except:
            return None

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
            data = prep(data)

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

    def n_classes(self):
        return len(self.classes)


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
                 n_max_len=200,
                 is_stk_processed=False
                 ):
        """
        :param root:
        :param suffix:
        :param back_mode: ['STK', 'S5']
        :param coor_mode: ['ABS', 'REL']
        :param n_max_len:
        :param is_stk_processed: 数据集是否已处理成 STK 格式
        """
        print(f'diffusion dataset, from: {root}')

        self.back_mode = back_mode
        self.coor_mode = coor_mode
        self.n_max_len = n_max_len
        self.is_stk_processed = is_stk_processed

        self.datapath = du.get_allfiles(root, suffix)

        print(f'number of instance all: {len(self.datapath)}')

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]

        if self.back_mode == 'STK':
            if self.is_stk_processed:
                sketch_cube = np.loadtxt(fn, delimiter=',')
                sketch_cube = sketch_cube.reshape([global_defs.n_stk, global_defs.n_stk_pnt, 2])
            else:
                sketch_cube = prep(fn)
            mask = 0
        elif self.back_mode == 'S5':
            sketch_cube, mask = dc.sketch_file_to_s5(fn, self.n_max_len, self.coor_mode)
        else:
            raise TypeError('error back mode')

        return sketch_cube, mask

    def __len__(self):
        return len(self.datapath)


class RetrievalDataset(Dataset):
    def __init__(self,
                 root,
                 data_mode='train',
                 max_seq_length=1200,
                 image_size=(224, 224),
                 back_mode='S5',
                 test_ratio=0.2,
                 is_shuffle=True
                 ):
        """
        图片通过将矢量图转化获得
        读取的可以是 txt，svg
        :param root:
        :param data_mode: ['train', 'test']
        :param max_seq_length:
        :param back_mode:
        :param test_ratio: 测试集占比
        :param is_shuffle: 是否随机划分训练集测试集

        root
        ├─ sketches
        │   ├─ airplane
        │   │   ├─ n02691156_58-1.svg
        │   │   ├─ n02691156_58-2.svg
        │   │   ├─ ...
        │   │
        │   ├─ alarm_clock
        │   │   ├─ n02694662_92-1.svg
        │   │   ├─ n02694662_92-2.svg
        │   │   ├─ ...
        │   ...
        │
        ├─ photos
        │   ├─ airplane
        │   │   ├─ n02691156_58.jpg
        │   │   ├─ n02691156_196.jpg
        │   │   ├─ ...
        │   │
        │   ├─ alarm_clock
        │   │   ├─ n02694662_92.jpg
        │   │   ├─ n02694662_166.jpg
        │   │   ├─ ...
        │   ...
        │

        """
        print(f'Retrieval dataset from: {root}')

        self.max_seq_length = max_seq_length
        self.back_mode = back_mode
        self.image_size = image_size
        self.data_mode = data_mode

        # 草图根目录
        sketch_root = os.path.join(root, 'sketches')

        # 草图类别
        classes = du.get_subdirs(sketch_root)

        self.sketch_photo_train = []
        self.sketch_photo_test = []
        # self.imgs_train = set()
        self.imgs_test = set()

        for c_class in classes:
            c_class_root = os.path.join(sketch_root, c_class)

            # 获取全部草图svg文件
            c_sketch_all = du.get_allfiles(c_class_root, 'txt')

            if is_shuffle:
                random.shuffle(c_sketch_all)

            n_c_sketch = len(c_sketch_all)
            test_idx = math.ceil(n_c_sketch * test_ratio)

            for idx, c_sketch in enumerate(c_sketch_all):
                # 获取图片文件名
                img_name = os.path.basename(c_sketch).split('-')[0] + '.jpg'
                img_path = os.path.join(root, 'photos', c_class, img_name)

                if idx < test_idx:
                    self.imgs_test.add(img_path)
                    self.sketch_photo_test.append((c_sketch, img_path))
                else:
                    # self.imgs_train.add(img_path)
                    self.sketch_photo_train.append((c_sketch, img_path))

        self.imgs_test = list(self.imgs_test)

        print(f'Training instance all: {len(self.sketch_photo_train)}')
        print(f'Testing instance all: {len(self.sketch_photo_test)}')
        print(f'Testing images all: {len(self.imgs_test)}')

    def __getitem__(self, index):
        if self.data_mode == 'train':
            sketch_root, photo_root = self.sketch_photo_train[index]
            c_index = 0

        elif self.data_mode == 'test':
            sketch_root, photo_root = self.sketch_photo_test[index]
            c_index = self.imgs_test.index(photo_root)

        elif self.data_mode == 'img':
            # 检索评价时用
            img_root = self.imgs_test[index]
            img_data = fr.img_read(img_root, self.image_size)

            return img_data, index

        else:
            raise TypeError('error dataset mode')

        if self.back_mode == 'STK':
            sketch_data = prep(sketch_root)
            mask = 0
        elif self.back_mode == 'S5':

            # vis_sketch_orig(sketch_root)

            sketch_data, mask = dc.sketch_file_to_s5(sketch_root, self.max_seq_length, 'ABS')

            # vis_s5_data(sketch_data)

        else:
            raise TypeError('error back mode')

        img_data = fr.img_read(photo_root, self.image_size)

        return sketch_data, mask, img_data, c_index

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.sketch_photo_train)
        elif self.data_mode == 'test':
            return len(self.sketch_photo_test)
        elif self.data_mode == 'img':
            return len(self.imgs_test)
        else:
            raise TypeError('error dataset mode')

    def train(self):
        self.data_mode = 'train'

    def eval(self):
        self.data_mode = 'test'

    def img(self):
        self.data_mode = 'img'


class SketchDatasetSeg(Dataset):
    def __init__(self, root=r'D:\document\DeepLearning\DataSet\sketch_seg\SketchSeg-150K', class_name='plane', data_mode='train', max_seq_length=256, image_size=(224, 224), return_mode='S5'):
        self.data_mode = data_mode

        npz_all = du.get_allfiles(root, 'npz')
        npz_train = None
        npz_test = None
        for c_npz in npz_all:
            if class_name in c_npz and 'train' in c_npz:
                npz_train = c_npz
            elif class_name in c_npz and 'test' in c_npz:
                npz_test = c_npz

        if npz_train is None or npz_test is None:
            raise ValueError('data not found')

        self.data_train = fr.npz_read(npz_train, 'arr_0', 'S5', 'REL', 200, is_back_seg=True)
        self.data_test = fr.npz_read(npz_test, 'arr_0', 'S5', 'REL', 200, is_back_seg=True)

        print(f'instance in training set: {len(self.data_train)}')
        print(f'instance in testing set: {len(self.data_test)}')

    def __getitem__(self, index):
        if self.data_mode == 'train':
            data = self.data_train
        elif self.data_mode == 'test':
            data = self.data_test
        else:
            raise TypeError('error dataset mode')

        data_cube, mask, seg_label = data[0][index], data[1][index], data[2][index]
        return data_cube, mask, seg_label

    def __len__(self):
        if self.data_mode == 'train':
            return len(self.data_train[0])
        elif self.data_mode == 'test':
            return len(self.data_test[0])
        else:
            raise TypeError('error dataset mode')

    def n_classes(self):
        return 57

    def train(self):
        self.data_mode = 'train'

    def eval(self):
        self.data_mode = 'test'


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
    # show_seg_imgs()


    pass



