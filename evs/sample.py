import numpy as np
from matplotlib import pyplot as plt
from functools import partial

import global_defs
from data_utils import filter as ft
from encoders import spline as sp
from data_utils import sketch_utils as du
from data_utils import vis
from data_utils import sketch_file_read as fr
from data_utils import data_convert as dc


def upsample_keep_dense(stroke, n_sample):
    """
    将给定笔划采样到制定点数，保持相对点密度不变
    :param stroke: [n, 2]
    :param n_sample:
    :return:
    """
    # 步骤1: 计算每一段的欧氏距离
    deltas = np.diff(stroke, axis=0)
    dists = np.linalg.norm(deltas, axis=1)

    # 步骤2: 累计长度（包括起点0）
    cumdist = np.concatenate([[0], np.cumsum(dists)])

    # 累积点数
    n_pnts = len(stroke)
    cumnpnt = np.linspace(1, n_pnts, n_pnts)

    # 插值获得插值点弧长参数
    samp_para = np.linspace(1, n_pnts, n_sample)
    interp_para = np.interp(samp_para, cumnpnt, cumdist)

    # 根据插值获得的弧长参数获得 x y
    interp_x = np.interp(interp_para, cumdist, stroke[:, 0])
    interp_y = np.interp(interp_para, cumdist, stroke[:, 1])

    interp_stk = np.hstack([interp_x[:, np.newaxis], interp_y[:, np.newaxis]])
    return interp_stk


def upsample_keep_dense_batched(stroke_list, n_sample):
    """
    每个笔划上采样到指定倍数的点
    :param stroke_list: 笔划列表
    :param n_sample: 上采样倍数
    :return:
    """

    assert isinstance(stroke_list, list)

    stroke_list_resampled = []
    for c_stk in stroke_list:
        stk_resampled = upsample_keep_dense(c_stk, n_sample)
        stroke_list_resampled.append(stk_resampled)

    return stroke_list_resampled


def preprocess_orig(sketch_root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, is_mix_proc=True, is_show_status=False, is_shuffle_stroke=False, is_order_stk=False) -> np.ndarray:
    """
    用于保持采样目的的预处理程序
    ---------------
    最初始的版本
    通过反复合并、拆分，使得笔划长度尽量相等
    可能导致笔划顺序乱掉
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

        # 删除长度过短的笔划
        sketch_data = ft.stroke_len_filter(sketch_data, 0.2)

        # 按笔划长度加权分配点数，假定共1000点
        all_len = []
        total_len = 0.0
        total_npnts = 1200
        for c_stk in sketch_data:
            c_len = du.stroke_length(c_stk)
            all_len.append(c_len)
            total_len += c_len

        sketch_data_resampled = []
        for c_len, c_stk in zip(all_len, sketch_data):
            c_pnts = int(total_npnts * (c_len / total_len))
            c_resample = upsample_keep_dense(c_stk, c_pnts)
            sketch_data_resampled.append(c_resample)

        sketch_data = sketch_data_resampled

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
        sketch_data = upsample_keep_dense_batched(sketch_data, n_stk_pnt)
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


def nonlinear_increasing_sampling(k, start=0, end=2 * np.pi, power=2):
    """
    在区间 [start, end] 上采样 k 个点，点之间间距递增。

    参数:
        k: 采样点个数
        start: 起始值，默认为 0
        end: 终止值，默认为 2π
        power: 控制递增速率（power > 1：递增更快）

    返回:
        长度为 k 的 numpy 数组，值在 [start, end] 范围内
    """
    # 生成均匀递增的基础序列 [0, 1]
    t = np.linspace(0, 1, k)

    # 应用幂函数使间距递增
    t_transformed = t ** power

    # 归一化映射到 [start, end]
    samples = start + (end - start) * (t_transformed - t_transformed[0]) / (t_transformed[-1] - t_transformed[0])

    return samples


def test():
    _n_points = 20

    _t = nonlinear_increasing_sampling(_n_points)
    _x = np.cos(_t)
    _y = np.sin(_t)

    _xy = np.hstack([_x.reshape(-1, 1), _y.reshape(-1, 1)])

    _new_xy = upsample_keep_dense(_xy, _n_points * 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列
    axes[0].plot(_x, _y)
    axes[0].scatter(_x, _y)
    axes[0].set_aspect('equal')

    axes[1].plot(_new_xy[:, 0], _new_xy[:, 1])
    axes[1].scatter(_new_xy[:, 0], _new_xy[:, 1])
    axes[1].set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    # dc.std_to_stk_batched(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean',
    #                    rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_presv_dnse_mix_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',
    #                    preprocess_orig)

    worker_func = partial(preprocess_orig,
                          is_mix_proc=False
                          )
    dc.s3_to_stk_batched(r'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean',
                       rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_presv_dnse_nomix_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',
                       worker_func)


    pass






