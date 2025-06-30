from svgpathtools import svg2paths
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import re
import xml.etree.ElementTree as ET

from data_utils import sketch_utils as du
import global_defs


def load_sketch_file(skh_file, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up, delimiter=','):
    """
    从草图文件中获取草图数据
    注意这里是直接读取文件存储的数据
    文件中存储的是绝对坐标，读取的就是绝对坐标。文件中存储的是相对坐标，读取的就是相对坐标。
    TU_Berlin 的 svg 文件中存储的是绝对坐标

    :param skh_file:
    :param pen_down:
    :param pen_up:
    :param delimiter:
    :return: [n, 3] (x, y, s)
    """
    suffix = os.path.splitext(skh_file)[1]

    if suffix == '.txt':
        sketch_data = np.loadtxt(skh_file, delimiter=delimiter)
    elif suffix == '.svg':
        sketch_data = svg_read(skh_file, pen_down, pen_up)
    else:
        raise TypeError('error file suffix')

    return sketch_data


def svg_read(svg_path, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up):
    """
    从 svg 文件读取草图
    :param svg_path:
    :param pen_down:
    :param pen_up:
    :return: [n, 3] (x, y, s)
    """

    def _filter_and_sort(_paths, _attributes):
        _filtered = []
        _pattern = re.compile(r'^svg_(\d+)$')

        for _idx, _attr in enumerate(_attributes):
            if isinstance(_attr, dict) and 'id' in _attr:
                _match = _pattern.fullmatch(_attr['id'])
                if _match:
                    num = int(_match.group(1))
                    _filtered.append((num, _idx))

        # 按 NUM 升序排序
        _filtered.sort()

        # 提取排序后的索引
        _sorted_indices = [_idx for _, _idx in _filtered]

        # 根据索引提取对应元素
        _sorted_attributes = [_attributes[_i] for _i in _sorted_indices]
        _sorted_paths = [_paths[_i] for _i in _sorted_indices]

        return _sorted_paths, _sorted_attributes

    paths, attributes = svg2paths(svg_path)
    paths, attributes = _filter_and_sort(paths, attributes)

    strokes = []
    for path in paths:
        if len(path) == 0:
            continue

        c_start = path[0].start
        c_stk = [(c_start.real, c_start.imag)]

        for segment in path:
            c_end = segment.end
            c_stk.append((c_end.real, c_end.imag))

        strokes.append(c_stk)

    stroke_list_np = []
    for c_stk in strokes:
        c_stk = np.array(c_stk)
        n = len(c_stk)
        ones_col = np.full((n, 1), pen_down, dtype=c_stk.dtype)
        ones_col[-1, 0] = pen_up
        c_stk = np.hstack((c_stk, ones_col))

        stroke_list_np.append(c_stk)

    stroke_list_np = np.vstack(stroke_list_np)

    return stroke_list_np


def npz_read(npz_root, data_mode='train', back_mode='STD', coor_mode='ABS', max_len=200, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up, is_back_seg=False):
    """
    读取 npz 文件中的草图，读取后的草图已归一化
    这里默认将 npz 文件中存储的数据视为相对坐标，因为 QuickDraw 数据集中的 npz 文件中存储的是相对坐标
    如果您的 npz 文件中存储绝对坐标，请修改

    :param npz_root:
    :param data_mode: ['train', 'test', 'valid']
    :param back_mode: ['STD', 'S5']
        'STD': [n, 3] (x, y, s)
        'S5': data: [N, 5], mask: [N, ], N = max_len + 2, 因为首尾要加两个标志位
    :param coor_mode: ['ABS', 'REL']
        'ABS': 绝对坐标
        'REL': 相对坐标
    :param max_len: S5 模式下的最长长度
    :param pen_down: quickdraw 中为 0
    :param pen_up: quickdraw 中为 1
    :param is_back_seg:
    :return:
    """

    data_all = np.load(str(npz_root), encoding='latin1', allow_pickle=True)
    dataset = data_all[data_mode]

    data = []
    mask = []
    if is_back_seg:
        seg = []

    for raw_data in dataset:
        try:
            # 获得绝对坐标
            xy_data = raw_data[:, :2]
            xy_data = np.cumsum(xy_data, axis=0)
            raw_data[:, :2] = xy_data

            if back_mode == 'S5':
                if len(raw_data) > max_len:
                    # 大于最大长度则从末尾截断
                    raw_data = raw_data[:max_len, :]

                if is_back_seg:
                    raw_data = torch.from_numpy(raw_data)
                    raw_seg = raw_data[:, 3:].max(1)[1]
                    raw_data = raw_data.numpy()

                # 归一化到 [-1, 1]
                raw_data = du.sketch_std(raw_data.astype(np.float32))

                # 转换为相对坐标
                if coor_mode == 'REL':
                    xy_data = raw_data[:, :2]
                    xy_data[1:] = xy_data[1:] - xy_data[:-1]
                    raw_data[:, :2] = xy_data

                # 提取到 data_cube 和 mask
                c_data = torch.zeros(max_len + 2, 5)
                c_mask = torch.zeros(max_len + 2)

                raw_data = torch.from_numpy(raw_data)
                len_raw_data = len(raw_data)

                # xy
                c_data[1:len_raw_data + 1, :2] = raw_data[:, :2]
                # $p_1$
                c_data[1:len_raw_data + 1, 2] = 1 - raw_data[:, 2]
                # $p_2$
                c_data[1:len_raw_data + 1, 3] = raw_data[:, 2]
                # $p_3$
                c_data[len_raw_data + 1:, 4] = 1

                if is_back_seg:
                    c_seg = torch.zeros(max_len + 2)
                    c_seg[1:len_raw_data + 1] = raw_seg

                # Mask is on until end of sequence
                c_mask[:len_raw_data + 1] = 1

                data.append(c_data)
                mask.append(c_mask)

                if is_back_seg:
                    seg.append(c_seg)

            elif back_mode == 'STD':
                """
                TODO: 需要在此添加相关代码返回分割标签
                """
                # 归一化到 [-1, 1]
                raw_data = du.sketch_std(raw_data.astype(np.float32))

                # 替换抬笔落笔标志位
                state = raw_data[:, 2]
                state = np.where(state == 0, pen_down, pen_up)
                raw_data[:, 2] = state

                # 转换为相对坐标
                if coor_mode == 'REL':
                    xy_data = raw_data[:, :2]
                    xy_data[1:] = xy_data[1:] - xy_data[:-1]
                    raw_data[:, :2] = xy_data

                data.append(raw_data)

            else:
                raise TypeError('error back mode')

        except:
            continue

    if is_back_seg:
        return data, mask, seg

    else:
        return data, mask


def img_read(img_root, img_size=(224, 224)):
    """
    从图片读取数据，返回包含数据的 tensor
    :param img_root:
    :param img_size:
    :return: [channel, width, height]
    """
    image = Image.open(img_root).convert("RGB")  # 确保是 RGB 模式

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    tensor_image = transform(image)

    return tensor_image







