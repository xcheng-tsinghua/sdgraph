import random

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from svgpathtools import svg2paths2
import warnings
import shutil
import math
from tqdm import tqdm
import requests
import torch
from matplotlib.collections import LineCollection
from torchvision import transforms
from PIL import Image
import cv2
import json
from multiprocessing import Pool
from functools import partial
from scipy.interpolate import CubicSpline

import global_defs
import encoders.spline as sp


def n_sketch_pnt(sketch) -> int:
    """
    返回草图中的点数
    :param sketch:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        return len(sketch)

    elif isinstance(sketch, list):
        pnt_count = 0
        for c_stk in sketch:
            pnt_count += len(c_stk)

        return pnt_count

    elif isinstance(sketch, str):
        sketch_data = np.loadtxt(sketch, delimiter=',')
        return len(sketch_data)

    else:
        raise TypeError('Unknown sketch type')


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    # 确定矩阵的大小（假设最大值为5，因此矩阵大小为6x6）
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    # 遍历 list1 和 list2 并更新矩阵
    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    # 使用 Matplotlib 可视化矩阵
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    try:
        plt.savefig(save_name)
    except:
        warnings.warn('can not save confusion matrix, for save path is not exist: ', save_name)
    plt.close()


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:

            if suffix is not None:
                if file.split('.')[-1] == suffix:
                    if filename_only:
                        current_filepath = file
                    else:
                        current_filepath = str(os.path.join(root, file))
                    filepath_all.append(current_filepath)

            else:
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


def translate_class_folder(root=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt\train'):
    """
    root
    ├─ bearing (folder)
    ├─ bolt (folder)
    ├─ ...
    └─ washer (folder)

    """
    terms_mapping = {
        '垫圈': 'Washer',
        '堵头': 'Plug',
        '带轮': 'Pulley',
        '弹簧': 'Spring',
        '挡圈': 'Retaining_ring',
        '接头': 'Joint',
        '法兰': 'Flange',
        '涡轮': 'Turbine',
        '脚轮': 'Caster',
        '螺柱': 'Stud',
        '螺栓': 'Bolt',
        '螺母': 'Nut',
        '螺钉': 'Screw',
        '衬套': 'Bushing',
        '轴承': 'Bearing',
        '铆钉': 'Rivet',
        '链轮': 'Sprocket',
        '销': 'Pin',
        '键': 'Key',
        '间隔器': 'Spacer',
        '阀门': 'Valve',
        '风扇': 'Fan',
        '齿轮': 'Gear'
    }

    sub_dirs = get_subdirs(root)

    for c_name in sub_dirs:
        new_name = terms_mapping[c_name]
        os.rename(os.path.join(root, c_name), os.path.join(root, new_name))


def sketch_std(sketch):
    """
    将草图质心移动到原点，范围归一化为 [-1, 1]^2
    :param sketch: [n_point, s]
    :return: 输入和输出类型相同
    """
    def _mean_coor_and_dist(_sketch_np):
        """

        :param _sketch_np: n*2 的 numpy 数组，表示草图
        :return:
        """
        _mean_coor = np.mean(_sketch_np, axis=0)
        _coordinates = _sketch_np - _mean_coor  # 实测是否加expand_dims效果一样
        _dist = np.max(np.sqrt(np.sum(_coordinates ** 2, axis=1)), 0)

        if _dist < 1e-5:
            raise ValueError('too small sketch scale')

        return _mean_coor, _dist

    def _move_scale_proc(_sketch_np, _mean_coor, _dist):
        _sketch_np = _sketch_np - _mean_coor  # 实测是否加expand_dims效果一样
        _sketch_np = _sketch_np / _dist

        return _sketch_np

    if len(sketch) == 0:
        raise ValueError('invalid stroke occurred, which contained zero points')

    if isinstance(sketch, np.ndarray):
        coordinates = sketch[:, :2]

        mean_coor, dist = _mean_coor_and_dist(coordinates)
        coordinates = _move_scale_proc(coordinates, mean_coor, dist)

        sketch[:, :2] = coordinates
        return sketch

    elif isinstance(sketch, list) and isinstance(sketch[0], np.ndarray):
        # 草图表示为sketch_list
        sketch_np = np.vstack(sketch)
        assert sketch_np.shape[1] == 2

        mean_coor, dist = _mean_coor_and_dist(sketch_np)

        stroke_list_new = []

        for c_stk in sketch:
            stk_new = _move_scale_proc(c_stk, mean_coor, dist)
            stroke_list_new.append(stk_new)

        return stroke_list_new

    else:
        raise TypeError('error sketch type')


def svg_read(svg_path, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up):
    """
    从 svg 文件读取草图
    :param svg_path:
    :param pen_down:
    :param pen_up:
    :return: [n, 3] (x, y, s)
    """
    paths, attributes, svg_attributes = svg2paths2(svg_path)

    strokes = []
    for path, attr in zip(paths, attributes):
        if len(path) == 0:
            continue

        # 分割子路径（处理M/m移动命令）
        subpaths = []
        current_subpath = []

        for segment in path:
            if segment.start != (current_subpath[-1].end if current_subpath else None):
                if current_subpath:
                    subpaths.append(current_subpath)
                current_subpath = []
            current_subpath.append(segment)

        if current_subpath:
            subpaths.append(current_subpath)

        # 处理每个子路径
        for subpath in subpaths:
            points = []
            # 添加第一个线段的起点
            points.append((subpath[0].start.real, subpath[0].start.imag))

            # 添加所有线段的终点
            for segment in subpath:
                points.append((segment.end.real, segment.end.imag))

            strokes.append(points)

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


def svg_to_txt(svg_path, txt_path, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up, delimiter=','):
    svg_data = svg_read(svg_path, pen_down, pen_up)
    # np.savetxt(txt_path, svg_data, fmt="%.5f", delimiter=delimiter)
    np.savetxt(txt_path, svg_data, delimiter=delimiter)

    # paths, attributes, svg_attributes = svg2paths2(svg_path)
    # strokes = []
    #
    # for path, attr in zip(paths, attributes):
    #     if len(path) == 0:
    #         continue
    #
    #     # 分割子路径（处理M/m移动命令）
    #     subpaths = []
    #     current_subpath = []
    #
    #     for segment in path:
    #         if segment.start != (current_subpath[-1].end if current_subpath else None):
    #             if current_subpath:
    #                 subpaths.append(current_subpath)
    #             current_subpath = []
    #         current_subpath.append(segment)
    #
    #     if current_subpath:
    #         subpaths.append(current_subpath)
    #
    #     # 处理每个子路径
    #     for subpath in subpaths:
    #         points = []
    #         # 添加第一个线段的起点
    #         points.append((subpath[0].start.real, subpath[0].start.imag))
    #
    #         # 添加所有线段的终点
    #         for segment in subpath:
    #             points.append((segment.end.real, segment.end.imag))
    #
    #         strokes.append(points)
    #
    # # for c_stk in strokes:
    # #     c_stk = np.array(c_stk)
    # #     plt.plot(c_stk[:, 0], -c_stk[:, 1])
    # #
    # # plt.axis('equal')
    # # plt.show()
    #
    # with open(txt_path, 'w') as f:
    #     for stroke_idx, stroke in enumerate(strokes):
    #         for i, (x, y) in enumerate(stroke):
    #             # 笔划状态判断（当前笔划的最后一个点标记s=0）
    #             s = 0 if (i == len(stroke) - 1) and (stroke_idx != len(strokes) - 1) else 1
    #
    #             # 写入文件，保留3位小数
    #             f.write(f"{round(x, 3):.3f},{round(y, 3):.3f},{s}\n")


def svg_to_txt_batched(source_dir, target_dir):
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

    files_all = get_allfiles(source_dir, 'svg')

    for c_file in tqdm(files_all, total=len(files_all)):
        try:
            svg_to_txt(c_file, c_file.replace(source_dir, target_dir).replace('svg', 'txt'))
        except:
            print(f'trans failure: {c_file}')


def create_tree_like(source_dir, target_dir):
    """
    在target_dir下创建与source_dir相同的目录层级
    :param source_dir:
    :param target_dir:
    :return:
    """
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)


def cls_distribute(source_dir, target_dir, test_rate=0.2):
    """
    将未分训练集和测试集的文件分开
    定位文件的路径如下：
    source_dir
    ├─ Bushes
    │   ├─0.obj
    │   ├─1.obj
    │   ...
    │
    ├─ Clamps
    │   ├─0.obj
    │   ├─1.obj
    │   ...
    │
    ...

    """
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    print('create dirs')
    create_tree_like(source_dir, os.path.join(target_dir, 'train'))
    create_tree_like(source_dir, os.path.join(target_dir, 'test'))

    # 复制文件
    # 获取全部类别
    classes_all = get_subdirs(source_dir)
    for c_class in tqdm(classes_all, total=len(classes_all)):

        c_class_dir = os.path.join(source_dir, c_class)
        c_files = get_allfiles(c_class_dir)

        n_files_all = len(c_files)
        n_files_test = math.ceil(n_files_all * test_rate)

        c_target_train = os.path.join(target_dir, 'train', c_class)
        c_target_test = os.path.join(target_dir, 'test', c_class)
        for i in range(n_files_all):
            base_name = os.path.basename(c_files[i])
            if i < n_files_test:
                shutil.copy(c_files[i], os.path.join(c_target_test, base_name))
            else:
                shutil.copy(c_files[i], os.path.join(c_target_train, base_name))


def quickdraw_download(target_dir):
    # 先提取全部类别
    class_file = r'C:\Users\ChengXi\Desktop\quickdraw.txt'
    class_all = []

    with open(class_file, 'r') as f:
        for c_line in f.readlines():
            c_line = c_line.strip()
            if 'full' in c_line:
                class_all.append(c_line.split('/')[-2])

    # print(class_all)
    # print(len(class_all))

    os.makedirs(target_dir, exist_ok=True)

    for category in tqdm(class_all, total=len(class_all)):
        url = f"https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category}"
        response = requests.get(url)
        with open(os.path.join(target_dir, category), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {category}")


def pen_updown_alt(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # 去除换行符并分割字段
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue  # 跳过格式不正确的行（可选）

            a, b, c, d = parts

            # 转换c字段：16→0，17→1
            new_c = '0' if c == '16' else '1'

            # 构造新行并写入（自动添加换行符）
            f_out.write(f"{a},{b},{new_c}\n")

def pen_updown_alt_batched(source_dir, target_dir, up_before=16, down_before=17, up_after=0, down_after=1):
    """
    修改源文件夹里的草图的up down指令
    :param source_dir:
    :param target_dir:
    :param up_before:
    :param down_before:
    :param up_after:
    :param down_after:
    :return:
    """
    # 先在target_dir创建与source_dir相同的目录层级
    print('clear dir:', target_dir)
    os.makedirs(target_dir, exist_ok=True)
    shutil.rmtree(target_dir)
    create_tree_like(source_dir, target_dir)

    files_all = get_allfiles(source_dir)

    for c_file in tqdm(files_all, total=len(files_all)):
        target_file = c_file.replace(source_dir, target_dir)
        pen_updown_alt(c_file, target_file)


# def equal_dist_resample(stroke: np.ndarray, resp_dist: float) -> np.ndarray:
#     """
#     等距采样,shortstraw中的，目前未调试成功
#     :param stroke:
#     :param resp_dist:
#     :return:
#     """
#     # 将 ndarray 转换为列表，每个元素为长度为2的 ndarray
#     stroke = [stroke[i, :2] for i in range(stroke.shape[0])]
#     resampled = [stroke[0]]  # 添加第一个点
#     accum_dist = 0.0  # 上一个采样点与当前点累积的距离
#     i = 1
#
#     while i < len(stroke):
#         c_dist = np.linalg.norm(stroke[i] - stroke[i - 1])
#
#         if accum_dist + c_dist >= resp_dist:
#             interp_pnt = stroke[i - 1] + (resp_dist - accum_dist) * (stroke[i] - stroke[i - 1])
#             resampled.append(interp_pnt)
#             stroke.insert(i, interp_pnt)
#             accum_dist = 0.0
#
#         else:
#             accum_dist += c_dist
#
#         i += 1
#
#     # 如果最后一个点与最后一个采样点之间的距离小于 0.5 * resp_dist，可额外加入最后一个点
#     # if np.linalg.norm(stroke_list[-1] - resampled[-1]) < 0.5 * resp_dist:
#     #     resampled.append(stroke_list[-1])
#
#     # 将采样后的点列表转换为 n×2 的 ndarray
#     return np.vstack(resampled)


def merge_point_group(stroke: np.ndarray, splits_raw: list, dist_thres: float) -> list:
    """
    将 stroke（n×2 的 ndarray）中的分割点 splits_raw 根据距离阈值 dist_thres 进行合并，
    返回每个分割组中间的索引列表。
    :param stroke: stroke points
    :param splits_raw: [(idx, straw), ...]
    :param dist_thres: 分割点组之间的距离必须大于该值
    """
    if len(splits_raw) == 0 or len(splits_raw) >= stroke.shape[0]:
        return []

    if len(splits_raw) == 1:
        return [splits_raw[0][0]]

    # 将索引转化为索引与起点到该索引点距离的元组，用于集群表示
    idx_and_arclen = []
    for c_split, c_straw in splits_raw:
        arc_len = stroke_length(stroke[:c_split + 1, :])

        idx_and_arclen.append((c_split, arc_len, c_straw))

    groups = []       # 最终的分组列表
    current_group = [idx_and_arclen[0]]  # 当前分组，初始放入第一个元组

    # 遍历数组，从第二个元素开始
    for tup in idx_and_arclen[1:]:
        # 如果当前元组的 dist 与当前分组最后一个元组的 dist 差值小于阈值，
        # 则认为它们属于同一组
        if tup[1] - current_group[-1][1] < dist_thres:
            current_group.append(tup)
        else:
            groups.append(current_group)
            current_group = [tup]  # 开启新一组

    # 添加最后一个分组
    groups.append(current_group)

    # 取各组的short straw最短的一个
    merged_idx = []
    for c_group in groups:
        if len(c_group) == 1:
            merged_idx.append(c_group[0][0])
        else:
            # idx_left = c_group[0][0]
            # idx_right = c_group[-1][0]
            min_tuple = min(c_group, key=lambda t: t[1])
            merged_idx.append(min_tuple[0])

    return merged_idx

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # c_overall_idx = 0
    # n_splits = len(splits_raw)
    #
    # # 第一个分割点作为初始组的第一个点
    # # 该数组存放分割组最左边的点，后续需要处理，将其处理成分割组中间的点
    # # split_idx = [splits_raw[0]]
    #
    # # 存储每个分割组中第一个和最后一个索引的元组
    # split_idx_start_end = [splits_raw[0]]
    # c_left = stroke[splits_raw[0]]
    #
    # for i in range(1, n_splits):
    #     # 计算 stroke 中两个分割点之间的欧氏距离
    #     # c_left = stroke[splits_raw[c_overall_idx]]
    #     c_split = stroke[splits_raw[i]]
    #
    #     if np.linalg.norm(c_left - c_split) > dist_thres:
    #         split_idx.append(splits_raw[i])
    #         c_overall_idx = i
    #         # 使用当前组中前一个采样点和上一个 split 点作为起始和结束索引
    #         # 有问题，如果某个分割组只有一个点的话
    #         split_idx_start_end.append((split_idx[-2], splits_raw[i - 1]))
    #
    # # 如果最后两个分割点距离小于等于阈值，则把最后一个点加入对应分割组
    # if n_splits > 2 and np.linalg.norm(stroke[splits_raw[-1]] - stroke[splits_raw[-2]]) <= dist_thres:
    #     split_idx_start_end.append((split_idx[-1], splits_raw[-1]))
    #
    # # 清空 split_idx 并用每个分割组中间的索引替代（整数除法，取整）
    # split_idx = []
    # for first, second in split_idx_start_end:
    #     split_idx.append((first + second) // 2)
    #
    # return split_idx


def split_continue(arr, breaks):
    # 总行数
    n = arr.shape[0]

    # 起止索引
    starts = [0] + breaks  # [0, 3, 6]
    ends = [b + 1 for b in breaks] + [n]  # [4, 7, 10]

    # 生成带重叠边界的子数组列表
    segments = [arr[s:e] for s, e in zip(starts, ends)]

    return segments


def short_straw_split(stroke: np.ndarray, resp_dist: float, filter_dist: float, thres: float, window_width: int, is_show_status=False, is_resample=True, is_split_once=False):
    """
    利用short straw进行角点分割，使用前必须将草图进行归一化至质心(0, 0)，范围[-1, 1]^2
    :param stroke: 单个笔划 [n, 2]
    :param resp_dist: 重采样间隔 [0, 1]
    :param is_split_once: 是否仅在一个角点进行分割？此时的分割点将在分割的两段笔划长度比较接近的点处分割
    :return:
    """
    thres_unify_judge = 1e-2
    # if 1 - np.max(stroke) >= -thres_unify_judge and np.min(stroke) + 1 >= -thres_unify_judge:
    #     pass
    # else:
    #     print(f'max: {np.max(stroke)}, and min: {np.min(stroke)}')

    # 辅助检测草图是否被归一化过
    assert 1 - np.max(stroke) >= -thres_unify_judge and np.min(stroke) + 1 >= -thres_unify_judge

    assert stroke.shape[1] == 2

    if is_show_status:
        all_straw = []

    splited_stk = []

    # Step 1: 重采样
    if is_resample:
        resample_stk = sp.uni_arclength_resample_strict_single(stroke, resp_dist)
    else:
        resample_stk = stroke

    # 如果重采样后的点数小于 window_width 的3倍，则返回原数组
    if len(resample_stk) < 3 * window_width:
        splited_stk.append(stroke)
        return splited_stk

    # Step 2: 计算 straw 值
    straw_and_idx = []
    straw_base = 0.0
    n_resample = len(resample_stk)
    half_window = window_width // 2

    straws_all = []
    straws_all.extend([0] * half_window)

    for i in range(half_window, n_resample - half_window):
        window_left = i - half_window
        window_right = i + half_window

        pnt_left = resample_stk[window_left]
        pnt_right = resample_stk[window_right]
        c_straw = np.linalg.norm(pnt_left - pnt_right)
        straw_and_idx.append((c_straw, i))

        straws_all.append(c_straw)

        if is_show_status:
            all_straw.append(c_straw)

        if c_straw > straw_base:
            straw_base = c_straw

        # if c_straw > 0.0225:
        #     asas = 0

    straws_all.extend([0] * half_window)

    # Step 3: 根据 straw 阈值确定角点
    straw_thres = straw_base * thres
    m_corners_idx = [(idx, straw) for (straw, idx) in straw_and_idx if straw < straw_thres]

    # Step 4: 合并过于接近的角点
    m_corners_idx = merge_point_group(resample_stk, m_corners_idx, filter_dist)

    # Step 5: 根据角点分割重采样后的笔画
    if is_split_once:
        n_pnts_all = len(resample_stk)
        # 获取每个分割点的分割距离
        corner_idx_diff = []
        for c_corner_idx in m_corners_idx:
            len_diff = abs((c_corner_idx + 1) - (n_pnts_all - c_corner_idx))
            corner_idx_diff.append((c_corner_idx, len_diff))

        min_idx = min(corner_idx_diff, key=lambda t: t[1])[0]
        former = resample_stk[0: min_idx + 1]
        later = resample_stk[min_idx:]
        splited_stk = [former, later]

    else:
        # splited_stk = np.split(resample_stk, m_corners_idx, axis=0)
        splited_stk = split_continue(resample_stk, m_corners_idx)

    if is_show_status:
        fig, axs = plt.subplots(1, 2)

        x = stroke[:, 0]
        y = -stroke[:, 1]
        c = straws_all

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, array=c, cmap='plasma', linewidth=3)
        axs[0].add_collection(lc)
        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(y.min(), y.max())
        fig.colorbar(lc, ax=axs[0], label='Color value (c)')

        # axs[0].plot(stroke[:, 0], -stroke[:, 1], c=straws_all)

        x = range(len(all_straw))
        axs[1].plot(x, all_straw)
        axs[1].plot([x[0], x[-1]], [straw_thres, straw_thres])

        plt.show()

    return splited_stk


def short_straw_split_once_until(sketch, resp_dist: float = 0.01, filter_dist: float = 0.1, thres: float = 0.9, window_width: int = 3, split_length: float = 0.2, is_print_split_status=False, is_resample=True):
    """
    单次在笔划的一个角点处分割
    TODO: 未完善
    :param sketch: list
    :param resp_dist:
    :param filter_dist:
    :param thres:
    :param window_width:
    :param split_length:
    :param is_print_split_status:
    :param is_resample:
    :return:
    """
    splited_sketch = []

    for c_stk in sketch:

        if stroke_length(c_stk) > split_length:
            splited_sketch += short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, is_print_split_status, is_resample)
        else:
            splited_sketch.append(c_stk)

    return splited_sketch


def sketch_short_straw_split(sketch, resp_dist: float = 0.01, filter_dist: float = 0.1, thres: float = 0.9, window_width: int = 3, split_length: float = 0.2, is_print_split_status=False, is_resample=True):
    """

    :param sketch:
    :param resp_dist:
    :param filter_dist: 相邻两个分割点之间的距离不小于该值
    :param thres: 当前点的 short_straw < thres * max(short_straw) 时，将该点判定为分割点
    :param window_width:
    :param split_length:
    :param is_print_split_status:
    :param is_resample:
    :return:
    """
    splited_sketch = []

    for c_stk in sketch:
        if stroke_length(c_stk) > split_length:
            splited_sketch += short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, is_print_split_status, is_resample)
        else:
            splited_sketch.append(c_stk)

    return splited_sketch


def stroke_length(stroke):
    if stroke.shape[0] < 2:
        return 0.0

    stroke = stroke[:, :2]
    diffs = np.diff(stroke, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    return np.sum(segment_lengths)


def stroke_length_var(stroke_list, is_use_stk_pnt=False):
    """
    计算草图中的笔划长度方差
    :param stroke_list:
    :param is_use_stk_pnt: 是否用笔划上点数代替长度加速计算
    :return:
    """
    stroke_length_all = []

    for c_stk in stroke_list:
        if is_use_stk_pnt:
            stroke_length_all.append(len(c_stk))
        else:
            stroke_length_all.append(stroke_length(c_stk))

    return np.var(stroke_length_all)


def single_split_(stroke_list: list) -> None:
    """
    将草图中最长的笔画对半分割成两个，这里的笔划长度等于点数，请注意
    :param stroke_list:
    :return:
    """
    if len(stroke_list) == 0:
        raise ValueError('input empty list')

    # Find the array with the maximum number of rows
    largest_idx = max(range(len(stroke_list)), key=lambda i: stroke_list[i].shape[0])
    largest_array = stroke_list[largest_idx]

    # Split the largest array into two halves
    split_point = largest_array.shape[0] // 2
    first_half = largest_array[:split_point + 1, :]
    second_half = largest_array[split_point:, :]

    # Replace the largest array with the two halves
    del stroke_list[largest_idx]
    stroke_list.append(first_half)
    stroke_list.append(second_half)


def single_split(stroke_list: list):
    sketch = stroke_list.copy()
    single_split_(sketch)
    return sketch


def split_stroke(stroke: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    将一个笔划拆分为两个近似对半的笔划。
    使用弧长累加的方法找到一个拆分点，使得
    拆分前的弧长接近总弧长的一半。
    拆分时，新笔划均包含拆分点以保证连续性。
    """
    if stroke.shape[0] < 2:
        # 如果只有一个点，无法拆分，直接返回原stroke两次
        raise ValueError('points in s stroke is too few')

    # 计算累积弧长，每个点对应从起点到该点的弧长
    diffs = np.diff(stroke, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cum_lengths[-1]
    half_length = total_length / 2.0

    # 找到第一个累积距离超过或等于half_length的位置
    idx = np.searchsorted(cum_lengths, half_length)
    # 为避免idx过于靠前或过于靠后，保证分割后两部分都有足够点数
    idx = max(1, min(idx, stroke.shape[0] - 1))

    # 用拆分点构造两个新笔划（共用拆分点）
    stroke1 = stroke[:idx + 1, :]
    stroke2 = stroke[idx:, :]
    return stroke1, stroke2


def sketch_split(sketch, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, delimiter=',') -> list:
    """
    根据标志符分割笔划，并去掉标志位
    :param sketch:
    :param pen_up:
    :param pen_down:
    :param delimiter:
    :return:
    """
    if isinstance(sketch, str):
        sketch = np.loadtxt(sketch, delimiter=delimiter)

    # 分割笔划
    sketch[-1, 2] = pen_down
    sketch = np.split(sketch[:, :2], np.where(sketch[:, 2] == pen_up)[0] + 1)

    return sketch


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
        dist_start = np.linalg.norm(end_point - stroke_list[i][0])
        if dist_start <= min_distance_start:
            min_distance_start = dist_start
            closest_idx_start = i

        dist_end = np.linalg.norm(end_point - stroke_list[i][-1])
        if dist_end <= min_distance_end:
            min_distance_end = dist_end
            closest_idx_end = i

    if min_distance_start <= min_distance_end:
        if min_distance_start == float('inf'):
            raise ValueError('inf dist occurred')

        return closest_idx_start, min_distance_start, True
    else:
        if min_distance_start == float('inf'):
            raise ValueError('inf dist occurred 2')

        return closest_idx_end, min_distance_end, False


def single_merge_(stroke_list, dist_gap, n_max_ita=200, max_merge_stk_len=float('inf')) -> bool:
    """
    将草图中最短的一个笔划合并到其他笔划
    注意：这里长度等于点数
    :param stroke_list:
    :param dist_gap: 若某个笔划距其它笔划的最近距离大于该值，不合并
    :param n_max_ita: 最大循环次数
    :param max_merge_stk_len: 合并的笔划长度大于该值，不执行合并
    :return: True: 成功合并一个短笔划， False: 合并失败
    """

    # 因距离其它笔画太远不合并的笔划
    stk_cannot_merge = []
    ita_count = 0

    while True:
        if len(stroke_list) < 2:
            if len(stk_cannot_merge) != 0:
                stroke_list.extend(stk_cannot_merge)

            return False

        # 防止迭代次数过多陷入死循环
        if ita_count > n_max_ita:
            if len(stk_cannot_merge) != 0:
                stroke_list.extend(stk_cannot_merge)

            return False
        else:
            ita_count += 1

        # 找到点数最少的笔划索引
        min_idx = min(range(len(stroke_list)), key=lambda i: len(stroke_list[i]))
        min_stroke = stroke_list[min_idx]

        min_start = min_stroke[0]  # 最短笔划的起始点
        min_end = min_stroke[-1]  # 最短笔划的起始点

        cidx_st, dist_st, is_ct_st_st = search_nearest_stroke(stroke_list, min_start, min_idx)
        cidx_ed, dist_ed, is_ct_st_ed = search_nearest_stroke(stroke_list, min_end, min_idx)

        # 如果最近的笔划距离其它笔画过远，不合并
        # 并将该最近的笔划放入不能合并列表，且将最短笔划从原始笔划列表删除
        if min(dist_st, dist_ed) > dist_gap:
            stk_cannot_merge.append(min_stroke)
            del stroke_list[min_idx]

        else:
            if stroke_length(min_stroke) <= max_merge_stk_len:

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
                    if np.linalg.norm(min_stroke[-1, :] - target_stk[0, :]) < 1e-6:  # 距离过近删除拼接点
                        min_stroke = min_stroke[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形2：起点到终点，this拼接在target后面：
                elif is_this_start and (not is_target_start):
                    if np.linalg.norm(target_stk[-1, :] - min_stroke[0, :]) < 1e-6:  # 距离过近删除拼接点
                        target_stk = target_stk[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                # 情形3：终点到起点，this拼接在target前面：
                elif (not is_this_start) and is_target_start:
                    if np.linalg.norm(min_stroke[-1, :] - target_stk[0, :]) < 1e-6:  # 距离过近删除拼接点
                        min_stroke = min_stroke[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形4：终点到终点，target不动，this调转后拼接在后面：
                elif (not is_this_start) and (not is_target_start):
                    min_stroke = np.flip(min_stroke, axis=0)
                    if np.linalg.norm(target_stk[-1, :] - min_stroke[0, :]) < 1e-6:  # 距离过近删除拼接点
                        target_stk = target_stk[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                else:
                    raise ValueError('error occurred in stroke merge start end judgement')

                del stroke_list[min_idx]  # 删除已合并的笔划

                # 加入之前不能合并的距其它笔划过远的笔划
                if len(stk_cannot_merge) != 0:
                    stroke_list.extend(stk_cannot_merge)

                return True

            else:
                if len(stk_cannot_merge) != 0:
                    stroke_list.extend(stk_cannot_merge)

                return False


def single_merge_dist_inc_(stroke_list, dist_begin=0.1, dist_inc=0.1, n_max_ita=200, max_merge_stk_len=float('inf')) -> bool:
    """
    以距离阈值递增的形式合并，保证一定能合并到
    :param stroke_list:
    :param dist_begin:
    :param dist_inc:
    :param n_max_ita:
    :param max_merge_stk_len:
    :return:
    """
    while not single_merge_(stroke_list, dist_begin):
        dist_begin += dist_inc


def single_merge(stroke_list, dist_gap, n_max_ita=200):
    sketch = stroke_list.copy()
    single_merge_(sketch, dist_gap, n_max_ita)
    return sketch


def stroke_merge_until(stroke_list, min_dist) -> list:
    """
    反复将stroke_list中的笔划合并，直到所有笔划间的距离大于min_dist为止
    这里笔划间的距离定义为笔划端点之间距离的最小值
    :param stroke_list:
    :param min_dist:
    :return:
    """

    def vis_sketch_list(strokes, show_dot=False, title=None):
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])

            if show_dot:
                plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        plt.axis("equal")
        plt.title(title)
        plt.show()

    new_list = stroke_list.copy()
    while True:
        is_merge_success = single_merge_(new_list, min_dist)

        # vis_sketch_list(sorted(new_list, key=lambda x: x.shape[0], reverse=True))

        if not is_merge_success:
            return new_list


def stroke_merge_number_until(stroke_list, max_n_stk, dist_gap_start=0.1):
    """
    反复将 stroke_list 中的笔划合并，直到笔划数等于 max_n_stk 为止
    如果 stroke_list 中笔划数小于等于 max_n_stk，直接返回原数组
    这里笔划间的距离定义为笔划端点之间距离的最小值
    :param stroke_list:
    :param max_n_stk:
    :param dist_gap_start:
    :return:
    """

    def vis_sketch_list(strokes, show_dot=False, title=None):
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])

            if show_dot:
                plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        plt.axis("equal")
        plt.title(title)
        plt.show()

    if len(stroke_list) <= max_n_stk:
        return stroke_list

    else:
        dist_inc = dist_gap_start
        new_list = stroke_list.copy()

        while True:
            while True:
                is_success = single_merge_(new_list, dist_gap_start)

                if len(new_list) <= max_n_stk:
                    return new_list

                if not is_success:
                    break

            dist_gap_start += dist_inc


def stroke_split_number_until(stroke_list, min_n_stk):
    """
    反复将 stroke_list 中最长笔划分割，直到笔划数等于 max_n_stk 为止，注意这里的笔划长度等于点数，请注意采样密度
    如果 stroke_list 中笔划数大于等于 min_n_stk，直接返回原数组
    :param stroke_list:
    :param min_n_stk:
    :return:
    """

    def vis_sketch_list(strokes, show_dot=False, title=None):
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])

            if show_dot:
                plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        plt.axis("equal")
        plt.title(title)
        plt.show()

    if len(stroke_list) >= min_n_stk:
        return stroke_list

    else:
        new_list = stroke_list.copy()
        while True:
            single_split_(new_list)

            if len(new_list) >= min_n_stk:
                return new_list


def short_stk_merge(stroke_list: list, max_stk_len: float, max_dist=0.2) -> list:
    """
    当草图中存在长度小于max_stk_len的笔划时，尝试将其合并
    :param stroke_list:
    :param max_stk_len:
    :param max_dist: 如果笔划端点之间的距离大于该值时，不考虑合并
    :return:
    """
    new_sketch = stroke_list.copy()

    while True:
        is_success = single_merge_(new_sketch, dist_gap=max_dist, max_merge_stk_len=max_stk_len)

        if not is_success:
            return new_sketch


class Rectangle(object):
    def __init__(self, min_x, max_x, min_y, max_y):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def area(self) -> float:
        width = abs(self.min_x - self.max_x)  # 水平边长
        height = abs(self.min_y - self.max_y)  # 垂直边长
        rect_area = width * height  # 矩形面积

        return rect_area

    def stk_mass_center_to_rect_boundary_dist(self, stroke) -> float:
        """
        计算笔划的质心到Rect边界的最小距离
        :param stroke:
        :return:
        """
        cx, cy = stroke.mean(axis=0)
        delta_x = max(self.min_x - cx, 0, cx - self.max_x)
        delta_y = max(self.min_y - cy, 0, cy - self.max_y)

        if delta_x == 0 and delta_y == 0:
            # 点在矩形内部：取到四边的最小距离
            dists = [cx - self.min_x, self.max_x - cx, cy - self.min_y, self.max_y - cy]
            distance = min(dists)
        else:
            # 点在矩形外部：欧氏距离
            distance = np.hypot(delta_x, delta_y)

        return distance

    def is_near(self, stroke, dist_thres) -> float:
        """
        判断 stroke 的质心是否离边界距离小于指定值
        :param stroke:
        :param dist_thres:
        :return:
        """
        dist = self.stk_mass_center_to_rect_boundary_dist(stroke)

        if dist < dist_thres:
            return True
        else:
            return False


def get_rect(stroke_list):
    """
    获取草图的包围盒
    :param stroke_list:
    :return: min_x, max_x, min_y, max_y
    """
    sketch = np.vstack(stroke_list)
    min_x = sketch[:, 0].min()
    max_x = sketch[:, 0].max()
    min_y = sketch[:, 1].min()
    max_y = sketch[:, 1].max()

    return Rectangle(min_x, max_x, min_y, max_y)


def stroke_list_to_sketch_cube(sketch_list):
    """
    将笔划转化为规则的 pytorch 的 Tensor
    有效位置为(x, y, 1)
    无效位置为(0, 0, -1)
    :param sketch_list:
    :return:
    """
    # sketch_cube = torch.full([global_defs.n_stk, global_defs.n_stk_pnt, 2], float('nan'), dtype=torch.float)
    # # sketch_cube = torch.full((global_defs.n_stk, global_defs.n_stk_pnt, 2), float('-inf'))
    # for i, c_stk in enumerate(sketch_list):
    #     n_cstk_pnt = len(c_stk)
    #     # sketch_mask[i, :n_cstk_pnt] = 1
    #
    #     sketch_cube[i, :n_cstk_pnt, :] = torch.from_numpy(c_stk)
    #
    # return sketch_cube

    sketch_cube = torch.zeros(global_defs.n_stk, global_defs.n_stk_pnt, 3, dtype=torch.float)
    for i, c_stk in enumerate(sketch_list):
        n_cstk_pnt = len(c_stk)

        ones_column = np.ones((n_cstk_pnt, 1))
        c_stk = np.hstack((c_stk, ones_column))

        sketch_cube[i, :n_cstk_pnt, :] = torch.from_numpy(c_stk)
        # sketch_cube[i, n_cstk_pnt:, 2] = -1.0

    return sketch_cube


def is_sketch_unified(stroke_list, thres=1e-5):

    sketch = np.vstack(stroke_list)

    if 1 - np.max(sketch) >= -thres and np.min(sketch) + 1 >= -thres:
        pass
    else:
        print(f'max: {np.max(sketch)}, and min: {np.min(sketch)}')


def sketch_file_to_s5(root, max_length, coor_mode='ABS', is_shuffle_stroke=False):
    """
    将草图转换为 S5 格式，(x, y, s1, s2, s3)
    默认存储绝对坐标
    :param root:
    :param max_length:
    :param coor_mode: ['ABS', 'REL'], 'ABS': absolute coordinate. 'REL': relative coordinate [(x,y), (△x, △y), (△x, △y), ...].
    :param is_shuffle_stroke: 是否打乱笔划
    :return:
    """
    if isinstance(root, str):
        file_suffix = Path(root).suffix
        if file_suffix == '.txt':
            data_raw = load_sketch_file(root)
        elif file_suffix == '.svg':
            data_raw = svg_read(root)
        else:
            raise TypeError('error suffix')
    else:
        raise TypeError('error root type')

    # 打乱笔划
    if is_shuffle_stroke:
        stroke_list = np.split(data_raw, np.where(data_raw[:, 2] == global_defs.pen_up)[0] + 1)[:-1]
        random.shuffle(stroke_list)
        data_raw = np.vstack(stroke_list)

    # 多于指定点数则进行采样
    n_point_raw = len(data_raw)
    if n_point_raw > max_length:
        data_raw = data_raw[:max_length, :]

        # choice = np.random.choice(n_point_raw, max_length, replace=True)
        # data_raw = data_raw[choice, :]

    # [n_points, 3]
    data_raw = sketch_std(data_raw)

    # plt.plot(data_raw[:, 0], data_raw[:, 1])
    # plt.show()

    # 相对坐标
    if coor_mode == 'REL':
        coordinate = data_raw[:, :2]
        coordinate[1:] = coordinate[1:] - coordinate[:-1]
        data_raw[:, :2] = coordinate

    elif coor_mode == 'ABS':
        # 无需处理
        pass

    else:
        raise TypeError('error coor mode')

    c_sketch_len = len(data_raw)
    data_raw = torch.from_numpy(data_raw)

    data_cube = torch.zeros(max_length, 5, dtype=torch.float)
    mask = torch.zeros(max_length, dtype=torch.float)

    data_cube[:c_sketch_len, :2] = data_raw[:, :2]
    data_cube[:c_sketch_len, 2] = data_raw[:, 2]
    data_cube[:c_sketch_len, 3] = 1 - data_raw[:, 2]
    data_cube[-1, 4] = 1

    mask[:c_sketch_len] = 1

    return data_cube, mask


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
                raw_data = sketch_std(raw_data.astype(np.float32))

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
                raw_data = sketch_std(raw_data.astype(np.float32))

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


def npz_statistic(root_npz=r'D:\document\DeepLearning\DataSet\quickdraw\raw'):
    """
    统计草图中的点数和该点数草图数量的统计值
    :param root_npz:
    :return:
    """
    npz_all = get_allfiles(root_npz, 'npz')
    skh_statistic = {}
    for c_npz in tqdm(npz_all, total=len(npz_all)):

        npz_train, _ = npz_read(c_npz, 'train')
        npz_test, _ = npz_read(c_npz, 'test')
        npz_valid, _ = npz_read(c_npz, 'valid')

        c_files_all = []
        c_files_all.extend(npz_train)
        c_files_all.extend(npz_test)
        c_files_all.extend(npz_valid)

        for c_skh in c_files_all:
            c_skh_pnt = len(c_skh)

            if c_skh_pnt not in skh_statistic.keys():
                skh_statistic[c_skh_pnt] = 1
            else:
                skh_statistic[c_skh_pnt] += 1

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(skh_statistic, f, ensure_ascii=False, indent=4)


def quickdraw_to_mgt(root_npz, root_target, delimiter=',', select=(1000, 100, 100), is_random_select=False):
    """
    该函数主要用于从 npz 文件中获取 MGT 数据集

    该函数根据 QuickDraw 的 npz 文件编写，主要特征如下：
    1. 存储相对坐标
    2. 加载的字典包含三个键，分别是 'train', 'test', 'valid'
    如果你的 npz 文件不符合以上要求，请修改

    将创建如下文件夹：
    root_target
    ├─ train
    │   └─ npz_name
    │       ├─ 1.txt
    │       ├─ 2.txt
    │       ├─ 3.txt
    │       ...
    │
    ├─ test
    │   └─ npz_name
    │       ├─ 1.txt
    │       ├─ 2.txt
    │       ├─ 3.txt
    │       ...
    │
    └─ valid
        └─ npz_name
            ├─ 1.txt
            ├─ 2.txt
            ├─ 3.txt
            ...

    :param root_npz:
    :param root_target:
    :param delimiter: 保存 txt 文件时的分隔符
    :param select: 从 [train, test, valid] 分支中抽取的草图数 (数量来自 MGT). = None 则不选取
    :param is_random_select: 是否随机选取
    :return:
    """

    def _get_n_pnt_near(_sketch_list, _select, _pnt_base=35):
        """
        从一个草图的 list 中选择指定数量的点数最靠近 _pnt_base 的草图
        :param _sketch_list:
        :param _select:
        :param _pnt_base:
        :return:
        """
        # 按行数与 pnt_base 的绝对差值排序
        sorted_arrays = sorted(_sketch_list, key=lambda arr: abs(arr.shape[0] - _pnt_base))

        # 返回前 select 个
        return sorted_arrays[:_select]

    # 先读取数据
    # print('load data')
    std_train = npz_read(root_npz, 'train')[0]
    std_test = npz_read(root_npz, 'test')[0]
    std_valid = npz_read(root_npz, 'valid')[0]

    if select is not None:
        sample_func = random.sample if is_random_select else _get_n_pnt_near

        std_train = sample_func(std_train, select[0])
        std_test = sample_func(std_test, select[1])
        std_valid = sample_func(std_valid, select[2])

    # 创建文件夹
    # print('create dirs')
    file_name = os.path.splitext(os.path.basename(root_npz))[0].replace('.', '_').replace(' ', '_')

    train_dir = os.path.join(root_target, 'train', file_name)
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(root_target, 'test', file_name)
    os.makedirs(test_dir, exist_ok=True)

    valid_dir = os.path.join(root_target, 'valid', file_name)
    os.makedirs(valid_dir, exist_ok=True)

    # 保存数据
    # print('save data')
    for idx, c_train in enumerate(std_train):
        c_train_filename = os.path.join(train_dir, f'{idx}.txt')
        np.savetxt(c_train_filename, c_train, delimiter=delimiter)

    for idx, c_test in enumerate(std_test):
        c_test_filename = os.path.join(test_dir, f'{idx}.txt')
        np.savetxt(c_test_filename, c_test, delimiter=delimiter)

    for idx, c_valid in enumerate(std_valid):
        c_valid_filename = os.path.join(valid_dir, f'{idx}.txt')
        np.savetxt(c_valid_filename, c_valid, delimiter=delimiter)


def quickdraw_to_mgt_batched(root_npz, root_target, is_random_select=True, workers=4):
    npz_all = get_allfiles(root_npz, 'npz')

    worker_func = partial(quickdraw_to_mgt, root_target=root_target, is_random_select=is_random_select)

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, npz_all),
            total=len(npz_all),
            desc='QuickDraw to MGT')
        )

    # for c_npz in tqdm(npz_all, total=len(npz_all)):
    #     quickdraw_to_mgt(c_npz, root_target, is_random_select=is_random_select)


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


def sketch_list_to_n3(sketch_list: list, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, positive_val=1, negative_val=-1):
    """
    将[(n0, 2), (n1, 2), ..., (n_N, 2)] 的stroke list转换为 [n_stk, n_stk_pnt, 3] 格式草图，第三维度表示点所在平面，不足的点用 PAD 补齐
    :param sketch_list:
    :param n_stk:
    :param n_stk_pnt:
    :param positive_val:
    :param negative_val:
    :return:
    """
    n3_cube = torch.zeros(n_stk, n_stk_pnt, 3)
    n_valid_stk = len(sketch_list)
    # last_pnt = torch.from_numpy(sketch_list[-1][-1])

    for idx, c_stk in enumerate(sketch_list):
        c_len = len(c_stk)
        c_torch_stk = torch.from_numpy(c_stk)
        n3_cube[idx, :c_len, :2] = c_torch_stk
        n3_cube[idx, :c_len, 2] = positive_val

        # n3_cube[idx, c_len:, :2] = c_torch_stk[-1]
        n3_cube[idx, c_len:, 2] = negative_val

    for idx in range(n_valid_stk, n_stk):
        # n3_cube[idx, :, :2] = last_pnt
        n3_cube[idx, :, 2] = negative_val

    return n3_cube


def sketch_list_to_tensor(sketch_list: list, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt):
    """
    将[(n0, 2), (n1, 2), ..., (n_N, 2)] 的stroke list转换为 [n_stk, n_stk_pnt, 2] 格式草图
    要求每个笔划中的点数相同
    :param sketch_list:
    :param n_stk:
    :param n_stk_pnt:
    :return:
    """
    n3_cube = torch.zeros(n_stk, n_stk_pnt, 2)

    for idx, c_stk in enumerate(sketch_list):
        c_torch_stk = torch.from_numpy(c_stk)
        n3_cube[idx, :, :] = c_torch_stk

    return n3_cube


def std_to_tensor_img(sketch, image_size=(224, 224), line_thickness=1, pen_up=global_defs.pen_up):
    """
    将 STD 草图转化为 Tensor 图片
    :param sketch: 文件路径或者加载好的 [n, 3] 草图
    :param image_size:
    :param line_thickness:
    :param pen_up:
    :return: list(image_size), 224, 224 为预训练的 vit 的图片大小
    """
    width, height = image_size

    if isinstance(sketch, str):
        points_with_state = load_sketch_file(sketch)

    elif isinstance(sketch, np.ndarray):
        points_with_state = sketch

    else:
        raise TypeError('error sketch type')

    # 1. 坐标归一化
    pts = np.array(points_with_state[:, :2], dtype=np.float32)
    states = np.array(points_with_state[:, 2], dtype=np.int32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diff_xy = max_xy - min_xy

    if np.allclose(diff_xy, 0):
        scale_x = scale_y = 1.0
    else:
        scale_x = (width - 1) / diff_xy[0] if diff_xy[0] > 0 else 1.0
        scale_y = (height - 1) / diff_xy[1] if diff_xy[1] > 0 else 1.0
    scale = min(scale_x, scale_y)

    pts_scaled = (pts - min_xy) * scale
    pts_int = np.round(pts_scaled).astype(np.int32)

    offset_x = (width - (diff_xy[0] * scale)) / 2 if diff_xy[0] > 0 else 0
    offset_y = (height - (diff_xy[1] * scale)) / 2 if diff_xy[1] > 0 else 0
    pts_int[:, 0] += int(round(offset_x))
    pts_int[:, 1] += int(round(offset_y))

    # 2. 创建白色画布
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 3. 笔划切分
    split_indices = np.where(states == pen_up)[0] + 1  # 下一个点是新笔划，所以+1
    strokes = np.split(pts_int, split_indices)

    # 4. 绘制每条笔划
    for stroke in strokes:
        if len(stroke) >= 2:  # 至少2个点才能画线
            stroke = stroke.reshape(-1, 1, 2)
            cv2.polylines(img, [stroke], isClosed=False, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    # 5. 转为归一化float32 Tensor
    tensor_img = torch.from_numpy(img).float() / 255.0

    cv2.imwrite(r'C:\Users\ChengXi\Desktop\fig\out.jpg', img)

    return tensor_img


def vis_s5_data(sketch_data, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down):
    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.show()


def stk_extend(stk: np.ndarray, n_extend=2):
    """
    将笔划左右分别向前插值n_extend个点
    :param stk:
    :param n_extend:
    :return:
    """
    if len(stk) < 3:
        return stk

    n = stk.shape[0]
    # 原始参数 t：均匀 0,1,...,n-1
    t = np.arange(n)

    x = stk[:, 0]
    y = stk[:, 1]

    # 构造可外推的三次样条
    spline_x = CubicSpline(t, x, extrapolate=True)
    spline_y = CubicSpline(t, y, extrapolate=True)

    # 生成新的 t 取值：从 -n_extend...-1, 再 0...n-1, 再 n...n-1+n_extend
    t_ext_head = np.arange(-n_extend, 0, 1)
    t_ext_tail = np.arange(n, n + n_extend, 1)

    # 组合所有 t
    t_all = np.concatenate([t_ext_head, t, t_ext_tail])

    # 样条插值（包括外推部分）
    x_all = spline_x(t_all)
    y_all = spline_y(t_all)

    extended_pts = np.vstack([x_all, y_all]).T
    return extended_pts


def stk_extend_batched(stk_list, n_extend=2) -> list:
    stks_extended = []
    for c_stk in stk_list:
        stks_extended.append(stk_extend(c_stk, n_extend))

    return stks_extended


if __name__ == '__main__':
    # svg_to_txt_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin\sketches', r'D:\document\DeepLearning\DataSet\TU_Berlin_txt')
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin_txt', r'D:\document\DeepLearning\DataSet\TU_Berlin_std')
    # cls_distribute(r'D:\document\DeepLearning\DataSet\TU_Berlin_std', r'D:\document\DeepLearning\DataSet\TU_Berlin_std_cls')


    # --------------------- 草图标准化
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch_cad\sketch_txt', rf'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')
    # object_name = 'cup'
    # std_unify_batched(
    #     rf'D:/document/DeepLearning/DataSet/sketch_from_quickdraw/{object_name}',
    #     rf'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/{object_name}_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}'
    # )

    # std_unify_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt', rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # cls_distribute(rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',
    #                rf'D:\document\DeepLearning\DataSet\TU_Berlin_cls_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\cup\5126.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\motorbike\10722.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\camera\3285.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Rivet\6dcc4bc1223014b26abb72b3dee939a8_1.txt'
    #
    # short_straw_split_sketch(sketch_test, is_show_status=False)

    # cls_distribute(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt', r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt_cls')

    # test()
    # test_resample()

    # quickdraw_download(r'D:\document\DeepLearning\DataSet\quickdraw_all')

    # pen_updown_alt_batched(r'D:\document\DeepLearning\DataSet\sketch_cad\sketch_txt', r'D:\document\DeepLearning\DataSet\sketch_cad\new')

    # 'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls\\train\\armchair\\530.txt'
    # exsketch = r'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls\\train\\armchair\\530.txt'
    # asketch = pre_process_seg_only(exsketch)
    # tmp_vis_sketch_list(asketch, True)

    # all_sketches = get_allfiles(r'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls')
    # all_sketches = get_allfiles(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt')
    #
    # for c_skh in all_sketches:
    #     asketch = pre_process_seg_only(c_skh)
    #     tmp_vis_sketch_list(asketch, True)

    # svg_root = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\cup\5125.svg'
    # svg_to_txt(svg_root, '')


    # test_npz = r'D:\document\DeepLearning\DataSet\quickdraw\raw\bicycle.full.npz'
    #
    # npz_read(test_npz, 'train', 'S5')

    # atestfile = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\svg\apple\333.svg'
    # sk_data, asmdk = sketch_file_to_s5(atestfile, 200, coor_mode='REL')
    #
    # x_adta = []
    # y_adta = []
    # c_coora = np.array([0., 0.])
    #
    # for i in range(len(sk_data)):
    #     dir_asd = sk_data[i].numpy()[:2]
    #     c_coora = c_coora + dir_asd
    #
    #     x_adta.append(c_coora[0])
    #     y_adta.append(-c_coora[1])
    #
    # plt.axis('equal')
    # plt.plot(x_adta, y_adta)
    # plt.show()

    # npz_to_txt(r'D:\document\DeepLearning\DataSet\quickdraw\small\ear.full.npz', r'D:\document\DeepLearning\DataSet\quickdraw\small')

    # npz_statistic()
    # quickdraw_to_mgt_batched(r'D:\document\DeepLearning\DataSet\quickdraw\raw', r'D:\document\DeepLearning\DataSet\quickdraw\MGT\random', is_random_select=True)
    # svg_to_txt_batched(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketches_svg', r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketches')



    # cdata, cmask = sketch_file_to_s5(r'D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\test_dataset\\sketches\\airplane\\n02691156_2173-4.txt', 1200, is_shuffle_stroke=True)
    # vis_s5_data(cdata)


    # plt.plot(cdata[:, 0].numpy(), cdata[:, 1].numpy())
    # plt.show()

    points = np.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]])

    pts_extend = stk_extend(points, 2)
    plt.plot(pts_extend[:, 0], pts_extend[:, 1])
    plt.scatter(pts_extend[:, 0], pts_extend[:, 1])
    plt.show()

    pass




