import einops
import numpy as np
import warnings
import os
import shutil
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

import global_defs
from data_utils import filter as ft
from encoders import spline as sp
from data_utils import sketch_utils as du
from data_utils import vis


def preprocess_orig(sketch_root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, is_mix_proc=True, is_show_status=False, is_shuffle_stroke=False):
    """
    最初始的版本
    通过反复合并、拆分，使得笔划长度尽量相等
    :return: [n_stk, n_stk_pnt, xy]
    """
    try:
        # 读取草图数据
        if isinstance(sketch_root, str):
            # 读取草图数据
            sketch_data = du.load_sketch_file(sketch_root)

        elif isinstance(sketch_root, (np.ndarray, list)):
            sketch_data = sketch_root

        else:
            raise TypeError('error input sketch_root type')

        # 移动草图质心并缩放大小
        sketch_data = du.sketch_std(sketch_data)

        # 按点状态标志位分割笔划
        sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

        # 重采样，使点之间距离尽量相等
        sketch_data = sp.uni_arclength_resample_strict(sketch_data, 0.01)

        # 去掉点数过少的笔划
        sketch_data = ft.stk_pnt_num_filter(sketch_data, 5)

        # 笔划数大于指定数值，将长度较短的笔划连接到最近的笔划
        if len(sketch_data) > n_stk:
            while len(sketch_data) > n_stk:
                du.single_merge_dist_inc_(sketch_data, 0.1, 0.1)

        # 笔划数小于指定数值，拆分点数较多的笔划
        elif len(sketch_data) < n_stk:
            while len(sketch_data) < n_stk:
                du.single_split_(sketch_data)

        if len(sketch_data) != global_defs.n_stk:
            raise ValueError(f'error stroke number: {len(sketch_data)}')

        if is_mix_proc:
            # 不断拆分、合并笔划，使各笔划点数尽量接近
            n_ita = 0
            var_brfore = 999.999
            while True:

                du.single_merge_dist_inc_(sketch_data, 0.1, 0.1)
                du.single_split_(sketch_data)

                var_after = du.stroke_length_var(sketch_data, True)

                if var_brfore == var_after or n_ita > 150:
                    break
                else:
                    var_brfore = var_after
                    n_ita += 1

        if len(sketch_data) != n_stk:
            raise ValueError(f'error stroke number final: {len(sketch_data)}, file: {sketch_root}')

        # 将每个笔划重采样至指定点
        sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, n_stk_pnt)
        if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample', show_dot=True)

        # 转换成 Tensor. [n_stk, n_stk_pnt, 2]
        sketch_data = np.array(sketch_data)
        if is_shuffle_stroke:
            np.random.shuffle(sketch_data)

        return sketch_data

    except:
        print('error file read')
        return None


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


def preprocess(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False):
    """
    每个笔划的点数不同
    因此不同笔划的点密度可能不同
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :param is_show_status:
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

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after filter outlier')

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    if is_show_status: vis.vis_sketch_list(sketch_data, title='after unify')

    # 去掉相邻过近的点，需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.005)

    # vis.vis_sketch_list(sketch_data)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after merge')

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    sketch_data = du.sketch_std(sketch_data)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after delete too short stroke')

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample')

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, thres=0.95, split_length=1.2, is_print_split_status=False, is_resample=False)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after split')

    # 角点分割分割可能产生非常短的笔划，当存在小于指定长度的短笔画时，尝试合并
    sketch_data = du.short_stk_merge(sketch_data, 0.8)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after merge short')

    # 长笔划分割
    sketch_data = ft.stk_len_maximum_filter(sketch_data, 1.3)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after seg too long stroke')

    # 去掉无效笔划，包含点数小于等于1的时无效笔划
    sketch_data = ft.valid_stk_filter(sketch_data)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove too short')

    # 按指定间隔采样点
    # sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, 0.04)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample speci dist')

    # 使所有笔划的点数均为2的整数倍
    sketch_data = ft.stk_pnt_double_filter(sketch_data)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after strech to double')

    # 对于点数大于指定数值的笔划，直接截断
    sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove too long points')

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在，如果低于指定数值，将草图全部数值置为零，且label也需要置为零

    # if len(sketch_data) < 4:
    #     warnings.warn(f'occurred n_stk lower than 4, is {sketch_root}')

    sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留长度最长的前 global_def.n_stk 个笔划
    sketch_data = ft.top_stk_len_filter(sketch_data, global_defs.n_stk)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove more stroke')

    # tmp_vis_sketch_list(sketch_data)
    # vis.vis_sketch_list(sketch_data, True, sketch_root)

    sketch_data = du.stroke_list_to_sketch_cube(sketch_data)

    return sketch_data


def preprocess_just_pad(sketch_root: str, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False) -> list:
    """
    仅仅填充
    :param sketch_root:
    :param pen_up:
    :param pen_down:
    :param is_show_status:
    :return:
    """
    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉outlier
    # sketch_data = ft.outlier_filter(sketch_data, 0.05, 0.3, 0.1)

    # 对于点数大于指定数值的笔划，直接截断
    sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove too long points')

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在，如果低于指定数值，将草图全部数值置为零，且label也需要置为零
    # sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留长度最长的前 global_def.n_stk 个笔划
    sketch_data = ft.top_stk_len_filter(sketch_data, global_defs.n_stk)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove more stroke')

    # tmp_vis_sketch_list(sketch_data)
    # vis.vis_sketch_list(sketch_data, True, sketch_root)

    return sketch_data


def preprocess_outlier_resamp_seg(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False) -> list:
    """
    每个笔划的点数不同
    因此不同笔划的点密度可能不同
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :param is_show_status:
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

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after filter outlier')

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    if is_show_status: vis.vis_sketch_list(sketch_data, title='after unify')

    # 去掉相邻过近的点，需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.005)

    # vis.vis_sketch_list(sketch_data)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after merge')

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    sketch_data = du.sketch_std(sketch_data)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after delete too short stroke')

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample')

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, thres=0.95, split_length=1.2, is_print_split_status=False, is_resample=False)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after split')

    # 角点分割分割可能产生非常短的笔划，当存在小于指定长度的短笔画时，尝试合并
    sketch_data = du.short_stk_merge(sketch_data, 0.8)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after merge short')

    # 长笔划分割
    sketch_data = ft.stk_len_maximum_filter(sketch_data, 1.3)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after seg too long stroke')

    # 去掉无效笔划，包含点数小于等于1的时无效笔划
    sketch_data = ft.valid_stk_filter(sketch_data)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove too short')

    # 将笔划点数采样至指定值
    sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample speci dist')

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在，如果低于指定数值，将草图全部数值置为零，且label也需要置为零

    # if len(sketch_data) < 4:
    #     warnings.warn(f'occurred n_stk lower than 4, is {sketch_root}')

    sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留长度最长的前 global_def.n_stk 个笔划
    sketch_data = ft.top_stk_len_filter(sketch_data, global_defs.n_stk)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove more stroke')

    # tmp_vis_sketch_list(sketch_data)
    # vis.vis_sketch_list(sketch_data, True, sketch_root)

    return sketch_data


def preprocess_force_seg_merge(sketch_root, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False):
    """
    指定笔划数，如果笔划数较少，强制分割
    如果笔划数过多，强制合并，不管距离远近
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :param is_show_status:
    :return:
    """
    if isinstance(sketch_root, str):
        # 读取草图数据
        sketch_data = du.load_sketch_file(sketch_root)

    elif isinstance(sketch_root, (np.ndarray, list)):
        sketch_data = sketch_root

    else:
        raise TypeError('error input sketch_root type')

    # 移动草图质心并缩放大小
    sketch_data = du.sketch_std(sketch_data)

    # 按点章台标志位分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉outlier
    sketch_data = ft.outlier_filter(sketch_data, 0.05, 0.3, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after filter outlier', show_dot=True)

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    if is_show_status: vis.vis_sketch_list(sketch_data, title='after unify', show_dot=True)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.05)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.05)

    sketch_data = du.sketch_std(sketch_data)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after delete too short stroke', show_dot=True)

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample', show_dot=True)

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, thres=0.95, split_length=0.8, is_print_split_status=False, is_resample=False)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after corner split', show_dot=True)

    # 角点分割分割可能产生非常短的笔划，当存在小于指定长度的短笔画时，尝试合并
    sketch_data = du.short_stk_merge(sketch_data, 0.2, 0.2)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after merge short', show_dot=True)

    # 去掉无效笔划，包含点数小于等于1的时无效笔划
    sketch_data = ft.valid_stk_filter(sketch_data)

    # 删除长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.03)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after remove too short', show_dot=True)

    # 若笔划数小于指定值，返回False
    # if len(sketch_data) < 3:
    #     vis.vis_sketch_list(sketch_data, title=sketch_root)
    #     return False

    # 若笔划数大于指定值，强制合并到指定数值
    sketch_data = du.stroke_merge_number_until(sketch_data, global_defs.n_stk)

    # 若笔划数小于指定数值，不断分割较长的笔划
    sketch_data = du.stroke_split_number_until(sketch_data, global_defs.n_stk)

    # 将笔划点数采样至指定值
    sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)

    # vis.vis_sketch_list(sketch_data, title='final', show_dot=True)

    sketch_data = np.array(sketch_data)
    return sketch_data


def preprocess_split_merge_until(sketch_root, resp_dist: float = 0.05, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False):
    """
    反复合并、拆分，直到不满足要求

    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :param is_show_status:
    :return:
    """
    if isinstance(sketch_root, str):
        # 读取草图数据
        sketch_data = du.load_sketch_file(sketch_root)

    elif isinstance(sketch_root, (np.ndarray, list)):
        sketch_data = sketch_root

    else:
        raise TypeError('error input sketch_root type')

    # 移动草图质心并缩放大小
    sketch_data = du.sketch_std(sketch_data)

    # 按点状态标志位分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉outlier
    sketch_data = ft.outlier_filter(sketch_data, 0.1, 0.3, 0.1)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after filter outlier', show_dot=True)

    # 归一化
    sketch_data = du.sketch_std(sketch_data)
    if is_show_status: vis.vis_sketch_list(sketch_data, title='after unify', show_dot=True)

    # 合并过近的笔划
    sketch_data = du.stroke_merge_until(sketch_data, 0.1)

    # 删除点数过少的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.21)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after delete too short stroke', show_dot=True)

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample', show_dot=True)

    # 分割点数大于指定值的笔划
    while max(sketch_data, key=lambda a: a.shape[0]).shape[0] > global_defs.n_stk_pnt:
        du.single_split_(sketch_data)

    # 若笔划数大于指定值，去掉其中比较短的
    sketch_data.sort(key=lambda a: a.shape[0], reverse=True)
    sketch_data = sketch_data[:global_defs.n_stk]

    # 如果笔划数小于指定值，不断分割长笔划
    sketch_data = du.stroke_split_number_until(sketch_data, 3)
    # vis.vis_sketch_list(sketch_data, title='final', show_dot=True)

    # 转化为 N3 格式
    sketch_data = du.sketch_list_to_n3(sketch_data)

    # sketch_data = np.array(sketch_data)
    return sketch_data


def resample_stake(sketch_root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False):
    # 几乎不处理，仅仅通过归一化、重采样并堆叠后返回
    if isinstance(sketch_root, str):
        # 读取草图数据
        sketch_data = du.load_sketch_file(sketch_root)

    elif isinstance(sketch_root, (np.ndarray, list)):
        sketch_data = sketch_root

    else:
        raise TypeError('error input sketch_root type')

    # 移动草图质心并缩放大小
    sketch_data = du.sketch_std(sketch_data)

    # 按点状态标志位分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 若笔划数大于指定值，强制合并到指定数值
    sketch_data = du.stroke_merge_number_until(sketch_data, global_defs.n_stk)



    # 将每个笔划重采样至指定点
    sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, global_defs.n_stk_pnt)
    if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample', show_dot=True)

    # 整理成 [n_stk, n_stk_pnt, 2] 的 tensor
    sketch_data = du.sketch_list_to_tensor(sketch_data)

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


def std_to_stk_file(std_file, source_dir, target_dir, preprocess_func, delimiter):
    try:
        c_target_file = std_file.replace(source_dir, target_dir)

        target_skh_STK = preprocess_func(std_file)
        target_skh_STK = einops.rearrange(target_skh_STK, 's sp c -> (s sp) c')
        # target_skh_STK = target_skh_STK.numpy()

        if len(target_skh_STK) == global_defs.n_skh_pnt:
            np.savetxt(c_target_file, target_skh_STK, delimiter=delimiter)
        else:
            print(f'error occurred, skip file: {std_file}')
    except:
        print(f'error occurred, skip file: {std_file}')


def std_to_stk_batched(source_dir, target_dir, preprocess_func, delimiter=',', workers=4):
    """
    将 source_dir 内的 std 草图转化为 STK 草图
    std 草图：每行为 [x, y, s]，行数不固定
    将在 target_dir 内创建与 source_dir 相同的层级结构
    文件后缀为 .STK
    :param source_dir:
    :param target_dir:
    :param preprocess_func:
    :param delimiter:
    :param workers: 处理进程数
    :return:
    """
    # 在 target_dir 内创建与 source_dir 相同的文件夹层级结构
    print('create dirs')

    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)

    # 获得source_dir中的全部文件
    files_all = du.get_allfiles(source_dir, 'txt')

    worker_func = partial(std_to_stk_file,
                          source_dir=source_dir,
                          target_dir=target_dir,
                          preprocess_func=preprocess_func,
                          delimiter=delimiter
                          )

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, files_all),
            total=len(files_all),
            desc='QuickDraw to MGT')
        )

    # for c_file in tqdm(files_all, total=len(files_all)):
    #
    #     try:
    #         c_target_file = c_file.replace(source_dir, target_dir)
    #
    #         target_skh_STK = preprocess_func(c_file)
    #         target_skh_STK = einops.rearrange(target_skh_STK, 's sp c -> (s sp) c')
    #         target_skh_STK = target_skh_STK.numpy()
    #
    #         if len(target_skh_STK) == global_defs.n_skh_pnt:
    #             np.savetxt(c_target_file, target_skh_STK, delimiter=delimiter)
    #     except:
    #         print(f'error occurred, skip file: {c_file}')


def print_tree_with_counts(root_path, prefix=""):
    """
    递归打印目录结构及每个目录下的文件数（不含子目录）。

    Args:
        root_path (str): 要遍历的根目录路径
        prefix (str): 当前层级前缀，用于缩进
    """
    try:
        # 列出该目录下所有项
        entries = os.listdir(root_path)
    except PermissionError:
        print(f"{prefix}└── [权限不足] {os.path.basename(root_path)}/")
        return

    # 计算文件数（仅文件，不含子目录）
    file_count = sum(1 for e in entries if os.path.isfile(os.path.join(root_path, e)))
    dirname = os.path.basename(root_path) or root_path
    print(f"{prefix}└── {dirname}/ ({file_count} files)")

    # 仅保留子目录并排序
    subdirs = sorted([e for e in entries if os.path.isdir(os.path.join(root_path, e))])
    for i, sub in enumerate(subdirs):
        path = os.path.join(root_path, sub)
        # 对最后一个子目录使用不同的分支符号，以对齐树状图
        if i == len(subdirs) - 1:
            branch = "    "  # 最后一个子目录，下一层不再延续“│”
        else:
            branch = "│   "
        print_tree_with_counts(path, prefix + branch)


def find_nonstandard_leaf_dirs(root_path, expected_counts=(100, 1000)):
    """
    遍历 root_path 下的所有目录，找出叶子目录（无子目录）且文件数不在 expected_counts 中的目录并打印。

    Args:
        root_path (str): 顶层目录路径
        expected_counts (tuple of int): 合格的文件数，其他文件数都将被打印
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 如果没有子目录，则为叶子目录
        if not dirnames:
            # 统计直接文件数（忽略子目录中的文件）
            try:
                file_count = sum(
                    1 for f in filenames
                    if os.path.isfile(os.path.join(dirpath, f))
                )
            except PermissionError:
                print(f"权限不足，无法访问：{dirpath}")
                continue

            # 如果文件数不在预期范围内，则打印
            if file_count not in expected_counts:
                print(f"{dirpath} （{file_count} 文件）")


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

    # all_sketches = du.get_allfiles(r'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls')
    # all_sketches = du.get_allfiles(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt')
    # random.shuffle(all_sketches)
    #
    # for c_skh in all_sketches:
    #     # vis.vis_sketch_orig(c_skh, title=c_skh, show_dot=True, dot_gap=5)
    #     vis.vis_sketch_orig(c_skh, show_dot=True, dot_gap=5)
    #
    #     asketch = preprocess_force_seg_merge(c_skh)
    #     print(asketch.shape)
    #     vis.vis_unified_sketch_data(asketch.reshape([-1, 2]), title=c_skh, show_dot=True)

    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Key\\9ac633e7e85d75207de0f4d44d51f456_2.txt'  # merge
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Pin\\b9a6e6512939a09d26d8892ff84253eb_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Pulley\\587e77a22dc8e953acfa7422d8cbfa6a_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Washer\\c73f2caa3d402863f727891573d57e86_1.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Joint\\085e56b3e4e9720977dda64a98c2ca6b_3.txt'  # outlier
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Rivet\\405a1dc601df05bda836a1aabeb68fdd_5.txt'
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Screw\\c8aed3582b5fe107324abed1afd0111b_4.txt'  # after proc, n_stk < 4
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Spring\\6e15365c70999807dd07ff812a7f4095_1.txt'  # after proc, n_stk < 4
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Screw\\c8aed3582b5fe107324abed1afd0111b_13.txt'  # after proc, n_stk < 4
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\test\\Flange\\64448e0877e197d89ba815e4ae203ed1_1.txt'  # after proc, n_stk < 4
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Spring\\6e15365c70999807dd07ff812a7f4095_4.txt'  # after proc, n_stk < 4
    # thefile = r'D:\\document\\DeepLearning\\DataSet\\sketch_cad\\raw\\sketch_txt\\train\\Key\\8a072034d5e8756e48c48361660e5fde_4.txt'

    # thefile = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\svg\sword\16974.svg'
    # thefile = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\svg\cigarette\4262.svg'  # false read<path id="12" d="M447,253C447,253 NaN,NaN 447,256CNaN,NaN 446.9427050983125,253.05729490168753 447,253C447.0190983005625,252.98090169943748 449,254 449,254"/>
    #
    # vis.vis_sketch_orig(thefile, title='sketch_orig')
    #
    # asasasas = preprocess_force_seg_merge(thefile, is_show_status=True)
    # # asasasas = pre_process_equal_stkpnt(thefile)
    #
    # vis.vis_sketch_list(asasasas, title='last', show_dot=True)


    # thefile = r'D:\document\DeepLearning\DataSet\quickdraw\MGT\train\camouflage_full\92.txt'
    # vis.vis_sketch_orig(thefile, title='sketch_orig')
    # asketch = preprocess_orig(thefile)
    # vis.vis_sketch_list(asketch, True)

    # sk_dir = r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple'
    #
    # sk_all = du.get_allfiles(sk_dir, 'txt')
    #
    # for c_file in sk_all:
    #     vis.vis_sketch_orig(c_file, title='sketch_orig')
    #
    #     asasasas = preprocess_split_merge_until(c_file, is_show_status=False)
    #
    #     # vis.vis_sketch_list(asasasas, title='last', show_dot=True)

    # a_test_npz = r'D:\document\DeepLearning\DataSet\quickdraw\raw\ambulance.full.npz'
    # a_test_npz = r'D:\document\DeepLearning\DataSet\quickdraw\raw\tiger.full.npz'
    # figs_all, _ = du.npz_read(a_test_npz)
    #
    # for c_fig in figs_all:
    #     vis.vis_sketch_data(c_fig)
    #
    #     resample_stake(c_fig, is_show_status=True)

    # std_to_stk_batched(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\random', rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_random_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', preprocess_orig)
    # std_to_stk_batched(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean',
    #                    rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',
    #                    preprocess_orig)

    find_nonstandard_leaf_dirs(rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')


    # folder = r'D:\document\DeepLearning\DataSet\quickdraw\MGT_stk_9_stk_pnt_32'
    # find_nonstandard_leaf_dirs(folder)

    # preprocess_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e0aa70a1d95a7e426cc6522eeddaa713_3.txt')

    # std_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean\train\bat_full\349.txt', r'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk11_stkpnt32\1.txt', preprocess_orig, ',')

    pass

