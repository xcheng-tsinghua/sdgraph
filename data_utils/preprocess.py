import numpy as np
import warnings
import torch
import matplotlib.pyplot as plt

import global_defs
from data_utils import filter as ft
from encoders import spline as sp
from data_utils import sketch_utils as du
from data_utils import vis
from data_utils import sketch_file_read as fr


def stroke_list_padding_to_cube(sketch_list: list, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, positive_val=1, negative_val=-1):
    """
    将[(n0, 2), (n1, 2), ..., (n_N, 2)] 的stroke list转换为 [n_stk, n_stk_pnt, 3] 格式草图，第三维度表示点所在平面，不足的点用 PAD 补齐
    有效位置为 [x, y, positive_val], 无效位置为 [0, 0, negative_val]
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

        # 末端重合
        # n3_cube[idx, c_len:, :2] = c_torch_stk[-1]
        n3_cube[idx, c_len:, 2] = negative_val

    for idx in range(n_valid_stk, n_stk):
        # n3_cube[idx, :, :2] = last_pnt
        n3_cube[idx, :, 2] = negative_val

    return n3_cube


def preprocess_orig(sketch_root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, is_mix_proc=True, is_show_status=False, is_shuffle_stroke=False, is_order_stk=False) -> np.ndarray:
    """
    最初始的版本
    通过反复合并、拆分，使得笔划长度尽量相等
    :return: [n_stk, n_stk_pnt, xy]
    """
    try:
        # 读取草图数据
        if isinstance(sketch_root, str):
            # 读取草图数据
            sketch_data = fr.load_sketch_file(sketch_root)

        elif isinstance(sketch_root, (np.ndarray, list)):
            sketch_data = sketch_root

        else:
            raise TypeError('error input sketch_root type')

        # 移动草图质心并缩放大小
        sketch_data = du.sketch_std(sketch_data)

        # 按点状态标志位分割笔划
        sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

        # 删除点数小于某个点数的笔划，可能删除比较长但是点数较少的笔划
        # sketch_data = ft.stk_pnt_num_filter(sketch_data, 4)

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

        # 将每个笔划左右插值两个点，使笔划间存在一定的重合区域，减缓生成时笔划过于分散
        sketch_data = du.stk_extend_batched(sketch_data, 1)

        # 将每个笔划重采样至指定点
        sketch_data = sp.uni_arclength_resample_certain_pnts_batched(sketch_data, n_stk_pnt)
        if is_show_status: vis.vis_sketch_list(sketch_data, title='after resample', show_dot=True)

        # 排序笔划
        if is_order_stk:
            sketch_data = du.order_strokes(sketch_data)

        # 转换成 Tensor. [n_stk, n_stk_pnt, 2]
        sketch_data = np.array(sketch_data)
        if is_shuffle_stroke:
            np.random.shuffle(sketch_data)

        return sketch_data

    except:
        print('error file read')
        return None


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

    sketch_data = stroke_list_padding_to_cube(sketch_data)

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
        sketch_data = fr.load_sketch_file(sketch_root)

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
        sketch_data = fr.load_sketch_file(sketch_root)

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

    # 将长短不一的笔划补齐
    sketch_data = stroke_list_padding_to_cube(sketch_data)

    # sketch_data = np.array(sketch_data)
    return sketch_data


def resample_stake(sketch_root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=False):
    # 几乎不处理，仅仅通过归一化、重采样并堆叠后返回
    if isinstance(sketch_root, str):
        # 读取草图数据
        sketch_data = fr.load_sketch_file(sketch_root)

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
    sketch_data = np.array(sketch_data)

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

    # find_nonstandard_leaf_dirs(rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\bicycle.full.npz', r'D:\document\DeepLearning\DataSet\quickdraw\diffusion')


    # folder = r'D:\document\DeepLearning\DataSet\quickdraw\MGT_stk_9_stk_pnt_32'
    # find_nonstandard_leaf_dirs(folder)

    # preprocess_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e0aa70a1d95a7e426cc6522eeddaa713_3.txt')

    # std_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean\train\bat_full\349.txt', r'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk11_stkpnt32\1.txt', preprocess_orig, ',')

    pass

