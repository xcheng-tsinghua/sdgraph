import torch
import numpy as np
import warnings
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import global_defs
from data_utils import filter as ft
from encoders import spline as sp
from data_utils import sketch_utils as du
from data_utils import vis


def std_unify(std_root: str, min_pnt: int = global_defs.n_stk * 2, is_mix_proc: bool = True):
    """
    将单个 std 草图转化为 unified_std 草图
    :param std_root: [n, 4]
    :param min_pnt: 处理后点数低于该数值的草图剔除
    :param is_mix_proc: 是否反复进行笔划拆分、合并、实现笔划的长度尽量相等
    :return:
    """

    # 读取std草图数据
    sketch_data = np.loadtxt(std_root, delimiter=',')
    sketch_data[-1, 2] = global_defs.pen_down

    # 去掉点数过少的笔划
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 4)

    # 去掉笔划上距离过近的点
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= min_pnt:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{std_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉过短的笔划
    strokes = ft.stroke_len_filter(strokes, 0.1)

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
            strokes = du.single_merge(strokes)

    # 小于指定数值，拆分点数较多的笔划
    elif len(strokes) < n_stk:
        while len(strokes) < n_stk:
            strokes = du.single_split(strokes)

    if len(strokes) != global_defs.n_stk:
        raise ValueError(f'error stroke number: {len(strokes)}')

    if is_mix_proc:
        # 不断拆分、合并笔划，使各笔划点数尽量接近
        n_ita = 0
        while True:
            before_var = du.stroke_length_var(strokes)

            strokes = du.single_merge(strokes)
            strokes = du.single_split(strokes)

            after_var = du.stroke_length_var(strokes)

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
    将 source_dir 文件夹中的全部 txt 草图转化为 unified_std 草图，并保存到 target_dir
    :param source_dir:
    :param target_dir:
    :return:
    """
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    print('create dirs')
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)

    files_all = du.get_allfiles(source_dir)

    for c_file in tqdm(files_all, total=len(files_all)):
        try:
            sketch_data = std_unify(c_file, is_mix_proc)
        except:
            warnings.warn(f'error occurred when trans {c_file}, has skipped!')
            continue

        if sketch_data is not None:
            try:
                sketch_data = np.concatenate(sketch_data, axis=0)
                sketch_data = sketch_data[:, :2]

                transed_npnts = len(sketch_data)
                if transed_npnts == global_defs.n_stk * global_defs.n_stk_pnt:

                    target_save_path = c_file.replace(source_dir, target_dir)
                    np.savetxt(target_save_path, sketch_data, delimiter=',')
                else:
                    warnings.warn(f'current point number is {transed_npnts}, skip file trans: {c_file}')
            except:
                warnings.warn(f'error occurred when trans {c_file}, has skipped!')


# def short_straw_split_sketch(sketch_root: str, resp_dist: float = 0.01, filter_dist: float = 0.1, thres: float = 0.9, window_width: int = 3,
#                              pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=True) -> list:
#     """
#     利用short straw对彩图进行角点分割
#     :param sketch_root:
#     :param resp_dist:
#     :param filter_dist:
#     :param thres:
#     :param window_width:
#     :param is_show_status:
#     :return:
#     """
#
#     # 读取草图数据
#     sketch_data = np.loadtxt(sketch_root, delimiter=',')
#
#     # 移动草图质心与大小
#     sketch_data = du.sketch_std(sketch_data)
#
#     # 分割草图
#     sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)
#
#     # tmp_vis_sketch_list(sketch_data)
#
#     # 去掉点数过少的笔划
#     # sketch_data = sp.stk_pnt_num_filter(sketch_data, 5)
#
#     # 去掉笔划上距离过近的点
#     # sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.03)
#
#     # 去掉长度过短的笔划
#     # sketch_data = sp.stroke_len_filter(sketch_data, 0.07)
#
#     vis.vis_sketch_list(sketch_data)
#
#     if du.n_sketch_pnt(sketch_data) <= 20:
#         warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
#         return []
#
#     # 分割笔划
#     strokes_splited = []
#     for c_stk in sketch_data:
#         strokes_splited += du.short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, False)
#
#     vis.vis_sketch_list(strokes_splited)
#
#     # 去掉点数过少的笔划
#     # strokes_splited = sp.stk_pnt_num_filter(strokes_splited, 16)
#
#     # 去掉笔划上距离过近的点
#     # strokes_splited = sp.near_pnt_dist_filter(strokes_splited, 0.03)
#
#     # 去掉长度过短的笔划
#     # strokes_splited = sp.stroke_len_filter(strokes_splited, 0.07)
#
#     # 仅保留点数最多的前 global_def.n_stk 个笔划
#     strokes_splited = ft.stk_number_filter(strokes_splited, global_defs.n_stk)
#
#     # 每个笔划中的点数仅保留前 global_def.n_pnt 个
#     strokes_splited = ft.stk_pnt_filter(strokes_splited, global_defs.n_stk_pnt)
#
#     if is_show_status:
#         for s in strokes_splited:
#             plt.plot(s[:, 0], -s[:, 1])
#             plt.scatter(s[:, 0], -s[:, 1])
#         plt.show()
#
#     return strokes_splited


# def pre_process(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
#     """
#     TODO: 需考虑太长笔划和太短笔划不能放在一起的问题，必须进行分割
#     :param sketch_root:
#     :param resp_dist:
#     :param pen_up:
#     :param pen_down:
#     :return:
#     """
#     # 读取草图数据
#     sketch_data = np.loadtxt(sketch_root, delimiter=',')
#
#     # 移动草图质心与大小
#     sketch_data = du.sketch_std(sketch_data)
#
#     # 分割笔划
#     sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)
#
#     # 去掉相邻过近的点
#     # -----------------需要先归一化才可使用，不然单位不统一
#     sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.001)
#
#     # 重采样
#     # sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)
#
#     # 角点分割
#     sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, is_print_split_status=False)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # tmp_vis_sketch_list(sketch_data, True)
#
#     # 去掉无效笔划
#     # sketch_data = sp.valid_stk_filter(sketch_data)
#
#     # 长笔划分割
#     sketch_data = ft.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # tmp_vis_sketch_list(sketch_data)
#
#     # 去掉点数过少的笔划
#     sketch_data = ft.stk_pnt_num_filter(sketch_data, 8)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # 使所有笔划的点数均为2的整数倍
#     sketch_data = ft.stk_pnt_double_filter(sketch_data)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
#     sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
#     sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
#     sketch_data = ft.stk_number_filter(sketch_data, global_defs.n_stk)
#
#     if len(sketch_data) == 0:
#         print(f'occurred zero sketch: {sketch_root}')
#         return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]
#
#     # tmp_vis_sketch_list(sketch_data)
#     # tmp_vis_sketch_list(sketch_data, True)
#
#     return sketch_data


def pre_process_equal_stkpnt(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
    """
    每个笔划的点数相同
    因此不同笔划的点密度可能不同
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :return:
    """
    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉outlier
    sketch_data = ft.outlier_filter(sketch_data, 0.05, 0.3, 0.1)
    # vis.vis_sketch_list(sketch_data, title='after filter outlier')

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    # vis.vis_sketch_list(sketch_data, title='after unify')

    # 去掉相邻过近的点，需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.005)

    # vis.vis_sketch_list(sketch_data)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.1)

    # vis.vis_sketch_list(sketch_data, title='after merge')

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    # vis.vis_sketch_list(sketch_data, title='after delete too short stroke')

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # vis.vis_sketch_list(sketch_data, title='after resample')

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, split_length=1.2, is_print_split_status=False, is_resample=False)

    # vis.vis_sketch_list(sketch_data, title='after split')

    # 角点分割分割可能产生非常短的笔划，当存在小于指定长度的短笔画时，尝试合并
    sketch_data = du.short_stk_merge(sketch_data, 0.8)

    # vis.vis_sketch_list(sketch_data, title='after merge short')

    # 长笔划分割
    sketch_data = ft.stk_len_maximum_filter(sketch_data, 1.5)

    # vis.vis_sketch_list(sketch_data, title='after seg too long stroke')

    # 去掉无效笔划，包含点数小于等于1的时无效笔划
    sketch_data = ft.valid_stk_filter(sketch_data)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.05)

    # vis.vis_sketch_list(sketch_data, title='after remove too short')

    # 将笔划点数采样至指定值
    sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在，如果低于指定数值，将草图全部数值置为零，且label也需要置为零
    sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留长度最长的前 global_def.n_stk 个笔划
    sketch_data = ft.top_stk_len_filter(sketch_data, global_defs.n_stk)

    # tmp_vis_sketch_list(sketch_data)
    # vis.vis_sketch_list(sketch_data, True, sketch_root)

    return sketch_data


# def pre_process(sketch_root: str, resp_dist: float = 0.03, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
#     """
#     :param sketch_root:
#     :param resp_dist:
#     :param pen_up:
#     :param pen_down:
#     :return:
#     """
#     # 读取草图数据
#     sketch_data = np.loadtxt(sketch_root, delimiter=',')
#
#     # 移动草图质心与大小
#     sketch_data = du.sketch_std(sketch_data)
#
#     # 分割笔划
#     sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)
#
#     # 去掉相邻过近的点
#     # -----------------需要先归一化才可使用，不然单位不统一
#     sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.01)
#
#     # 合并过近的笔划
#     sketch_data = du.stroke_merge_until(sketch_data, 0.05)
#
#     # 删除长度过短的笔划
#     sketch_data = ft.stroke_len_filter(sketch_data, 0.1)
#
#     # 重采样
#     sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)
#
#     # 去掉长度过短的笔划
#     # sketch_data = ft.stroke_len_filter(sketch_data, 0.1)
#
#     # 角点分割
#     sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, split_length=0.5, is_print_split_status=False, is_resample=False)
#
#     # tmp_vis_sketch_list(sketch_data, True)
#
#     # 去掉无效笔划
#     # sketch_data = sp.valid_stk_filter(sketch_data)
#
#     # 长笔划分割
#     sketch_data = ft.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)
#
#     # tmp_vis_sketch_list(sketch_data)
#
#     # 去掉点数过少的笔划
#     sketch_data = ft.stk_pnt_num_filter(sketch_data, 8)
#
#     # 使所有笔划的点数均为2的整数倍
#     sketch_data = ft.stk_pnt_double_filter(sketch_data)
#
#     # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
#     sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)
#
#     # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
#     # 如果低于指定数值，将草图全部数值置为零
#     sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)
#
#     # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
#     sketch_data = ft.stk_number_filter(sketch_data, global_defs.n_stk)
#
#     # tmp_vis_sketch_list(sketch_data)
#     # vis.vis_sketch_list(sketch_data, True, sketch_root)
#
#     return sketch_data


def preprocess(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
    """
    每个笔划的点数不同
    因此不同笔划的点密度可能不同
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :return:
    """
    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉outlier
    sketch_data = ft.outlier_filter(sketch_data, 0.05, 0.3, 0.1)
    # vis.vis_sketch_list(sketch_data, title='after filter outlier')

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    # vis.vis_sketch_list(sketch_data, title='after unify')

    # 去掉相邻过近的点，需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.005)

    # vis.vis_sketch_list(sketch_data)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.1)

    # vis.vis_sketch_list(sketch_data, title='after merge')

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    # vis.vis_sketch_list(sketch_data, title='after delete too short stroke')

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # vis.vis_sketch_list(sketch_data, title='after resample')

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, split_length=1.2, is_print_split_status=False, is_resample=False)

    # vis.vis_sketch_list(sketch_data, title='after split')

    # 角点分割分割可能产生非常短的笔划，当存在小于指定长度的短笔画时，尝试合并
    sketch_data = du.short_stk_merge(sketch_data, 0.8)

    # vis.vis_sketch_list(sketch_data, title='after merge short')

    # 长笔划分割
    sketch_data = ft.stk_len_maximum_filter(sketch_data, 1.2)

    # vis.vis_sketch_list(sketch_data, title='after seg too long stroke')

    # 去掉无效笔划，包含点数小于等于1的时无效笔划
    sketch_data = ft.valid_stk_filter(sketch_data)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    # vis.vis_sketch_list(sketch_data, title='after remove too short')

    # 按指定间隔采样点
    # sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, 0.03)

    # 使所有笔划的点数均为2的整数倍
    sketch_data = ft.stk_pnt_double_filter(sketch_data)

    # 对于点数大于指定数值的笔划，直接截断
    sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在，如果低于指定数值，将草图全部数值置为零，且label也需要置为零

    if len(sketch_data) < 4:
        warnings.warn(f'occurred n_stk lower than 4, is {sketch_root}')

    sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留长度最长的前 global_def.n_stk 个笔划
    sketch_data = ft.top_stk_len_filter(sketch_data, global_defs.n_stk)

    # tmp_vis_sketch_list(sketch_data)
    # vis.vis_sketch_list(sketch_data, True, sketch_root)

    return sketch_data


def test_resample():
    sketch_root = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\motorbike\10722.txt'
    # sketch_root = sketch_test

    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')
    # sketch_data[-1, 2] = global_defs.pen_down

    # 去掉点数过少的笔划
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 5)

    # 去掉笔划上距离过近的点
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= 25:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉长度过短的笔划
    strokes = ft.stroke_len_filter(strokes, 0.07)

    # 重采样
    strokes_resampled = sp.uni_arclength_resample_strict(strokes, 0.1)

    for s in strokes_resampled:
        plt.plot(s[:, 0], -s[:, 1])
        plt.scatter(s[:, 0], -s[:, 1])

    plt.show()
    pass


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
    # all_sketches = du.get_allfiles(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt')
    # random.shuffle(all_sketches)
    #
    # for c_skh in all_sketches:
    #     vis.vis_sketch_orig(c_skh, title=c_skh)

        # asketch = pre_process(c_skh)
        # vis.vis_sketch_list(asketch, True)

    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Key\\9ac633e7e85d75207de0f4d44d51f456_2.txt'  # merge
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Pin\\b9a6e6512939a09d26d8892ff84253eb_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Pulley\\587e77a22dc8e953acfa7422d8cbfa6a_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Washer\\c73f2caa3d402863f727891573d57e86_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Joint\\085e56b3e4e9720977dda64a98c2ca6b_3.txt'  # outlier
    thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Rivet\\405a1dc601df05bda836a1aabeb68fdd_5.txt'

    vis.vis_sketch_orig(thefile)

    asasasas = pre_process_equal_stkpnt(thefile)

    vis.vis_sketch_list(asasasas, title='last')


    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Bearing\\17b0dd39d358ce217e7c76a8a20a40fe_6.txt'
    # asketch = pre_process(thefile)
    # vis.vis_sketch_list(asketch, True)



    pass

