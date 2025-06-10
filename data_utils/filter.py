import numpy as np
import matplotlib.pyplot as plt

import global_defs
from data_utils import sketch_utils as du


def stk_pnt_num_filter(sketch, min_point=5) -> list:
    """
    filter out strokes whose point number is below min_point
    :param sketch: (list) element: ndarray[n_point, 2]
    :param min_point: (int)
    :return:
    """
    # 如果是ndarray，说明笔划未分割
    if isinstance(sketch, np.ndarray):
        sketch = du.sketch_split(sketch)

    filtered_stk = []
    for c_stk in sketch:
        if len(c_stk) >= min_point:
            filtered_stk.append(c_stk)

    return filtered_stk


def stroke_point_dist_filter(stroke_list, min_dist=1e-3):
    """
    filter out stroke points whose dist to previous point is below min_dist
    :param stroke_list: (list) element: ndarray[n_point, 2]
    :param min_dist: (float)
    :return:
    """
    filtered_stk = []

    # -> c_stk: [n_stk_point, 4]
    for c_stk in stroke_list:

        filtered_points = [c_stk[0]]
        for point in c_stk[1:]:
            if np.linalg.norm(point[:2] - filtered_points[-1][:2]) >= min_dist:
                filtered_points.append(point)

        filtered_stk.append(np.array(filtered_points))

    return filtered_stk


def near_pnt_dist_filter(sketch, tol):
    """
    删除距离过近的点
    :param sketch: np.ndarray[n_pnt, 3]
    :param tol:
    :return:
    """
    valid_stks = []

    # 如果是ndarray，说明输入的是单个笔划
    if isinstance(sketch, np.ndarray):
        skh_length = len(sketch)

        valid_stks.append(sketch[0])
        prev_idx = 0
        for c_pnt_idx in range(1, skh_length):
            c_pnt = sketch[c_pnt_idx, :2]
            prev_pnt = sketch[prev_idx, :2]

            c_dist = np.linalg.norm(c_pnt - prev_pnt)
            if c_dist >= tol:
                valid_stks.append(sketch[c_pnt_idx])
                prev_idx = c_pnt_idx

        valid_stks = np.vstack(valid_stks)

    else:
        for c_stk in sketch:
            c_valid_stk_pnts = [c_stk[0]]
            prev_idx = 0
            for c_pnt_idx in range(1, len(c_stk)):
                c_pnt = c_stk[c_pnt_idx]
                prev_pnt = c_stk[prev_idx]

                c_dist = np.linalg.norm(c_pnt - prev_pnt)
                if c_dist > tol:
                    c_valid_stk_pnts.append(c_stk[c_pnt_idx])
                    prev_idx = c_pnt_idx

            valid_stks.append(np.vstack(c_valid_stk_pnts))

    return valid_stks


def stroke_num_filter(stroke_list, stroke_num):
    """
    get top stroke_num strokes by stroke points
    :param stroke_list:
    :param stroke_num:
    :return:
    """
    array_info = [(i, arr.shape[0]) for i, arr in enumerate(stroke_list)]

    # 按行数从大到小排序
    sorted_arrays = sorted(array_info, key=lambda x: x[1], reverse=True)

    # 提取前 m 个数组的索引
    top_m_indices = [idx for idx, _ in sorted_arrays[:stroke_num]]

    # 获取对应的 numpy 数组
    top_m_arrays = [stroke_list[idx] for idx in top_m_indices]

    return top_m_arrays


def stroke_len_filter(stroke_list, min_length=5e-2):
    """
    删除数组中长度小于 min_length 的笔划
    :param stroke_list:
    :param min_length:
    :return:
    """
    if isinstance(stroke_list, np.ndarray):
        stroke_list = du.sketch_split(stroke_list)

    filtered_stk = []

    for c_stk in stroke_list:
        c_stroke_length = du.stroke_length(c_stk)
        if c_stroke_length >= min_length:
            filtered_stk.append(c_stk)

    return filtered_stk


def stk_number_filter(sketch, n_stk=global_defs.n_stk):
    """
    保留笔划中点数较多的前n_stk个笔画
    :param sketch:
    :param n_stk:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        sketch = du.sketch_split(sketch)

    sketch = sorted(sketch, key=lambda arr: arr.shape[0], reverse=True)
    sketch = sketch[:n_stk]

    return sketch


def stk_pnt_filter(sketch, n_stk_pnt=global_defs.n_stk_pnt):
    """
    保留笔划中的前n_stk_pnt个点
    :param sketch:
    :param n_stk_pnt:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        sketch = du.sketch_split(sketch)

    filtered_sketch = []

    for c_stk in sketch:
        if len(c_stk) > n_stk_pnt:
            filtered_sketch.append(c_stk[:n_stk_pnt])
        else:
            filtered_sketch.append(c_stk)

    return filtered_sketch


def stk_pnt_double_filter(sketch):
    """
    确保每个笔划上的点数均为2的整数倍，
    若不符合要求，向前插值一个点
    :param sketch:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        sketch = du.sketch_split(sketch)

    filtered_sketch = []
    for c_stk in sketch:
        if len(c_stk) % 2 != 0:
            last = c_stk[-1]
            last_back = c_stk[-2]

            forward = 2 * last - last_back
            c_stk = np.vstack([c_stk, forward])

        #     print('触发插值')
        #     plt.scatter(c_stk[:-1, 0], -c_stk[:-1, 1])
        #     plt.scatter(c_stk[-1, 0], -c_stk[-1, 1], color='black')
        # else:
        #     plt.scatter(c_stk[:, 0], -c_stk[:, 1])

        filtered_sketch.append(c_stk)

    plt.show()
    return filtered_sketch


def stk_num_minimal_filter(sketch, n_stk_min):
    """
    保证草图中的笔划数不能小于指定值
    如果小于，sketch变为全为零数组
    :param sketch:
    :param n_stk_min:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        sketch = du.sketch_split(sketch)

    if len(sketch) < n_stk_min:
        sketch = []
        # warnings.warn(f'the number of strokes is lower than {n_stk_min}, all data is set to ZERO')

        for i in range(global_defs.n_stk):
            sketch.append(np.zeros((global_defs.n_stk_pnt, 2), dtype=np.float32))

    # while True:
    #     if len(sketch) >= n_stk_min:
    #         break
    #
    #     du.single_split_(sketch)

    return sketch


def valid_stk_filter(sketch: list):
    """
    某些笔划可能为空，即该位置的笔划为size=(0,)，需要删除
    另外有些笔划可能只有一个点，需要删除
    :param sketch:
    :return:
    """
    sketch_new = []

    for c_stk in sketch:
        if len(c_stk) >= 2:
            sketch_new.append(c_stk)

    return sketch_new


def stk_n_pnt_maximum_filter(sketch: list, n_pnt_max: int) -> list:
    """
    最长笔划中的点数必须小于 n_pnt_max，否则递归对半拆分最长的笔划
    :param sketch:
    :param n_pnt_max:
    :return:
    """
    strokes = sketch.copy()

    while True:
        max_stroke = max(strokes, key=lambda _stroke: _stroke.shape[0])

        if max_stroke.shape[0] <= n_pnt_max:
            break
        else:
            du.single_split_(strokes)

    return strokes

    # # 逐个检查是否存在超长笔划
    # need_check = True
    # while need_check:
    #     need_check = False  # 假设此次循环中无超长笔划
    #     for i, stroke in enumerate(strokes):
    #         if stroke_length(stroke) > stk_len_max:
    #             # 如果超长，对其拆分并替换原来的笔划
    #             stroke1, stroke2 = split_stroke(stroke)
    #             # 用拆分得到的两个笔划替换原来的笔划
    #             strokes.pop(i)
    #             strokes.insert(i, stroke2)
    #             strokes.insert(i, stroke1)
    #             # 标记需要再次检查，因为拆分后可能还是超长
    #             need_check = True
    #             # 跳出当前循环，重新开始遍历列表
    #             break
    #
    # return strokes


def top_stk_len_filter(sketch: list, n_stk_max: int) -> list:
    """
    取长度最长的n_stk_max个笔划
    :param sketch:
    :param n_stk_max:
    :return:
    """

    if len(sketch) <= n_stk_max:
        return sketch

    else:
        sorted_list = sorted(sketch, key=lambda arr: du.stroke_length(arr), reverse=True)
        return sorted_list[:n_stk_max]


def stk_len_maximum_filter(sketch: list, stk_len_max: float) -> list:
    """
    最长笔划的长度必须小于 stk_len_max，否则递归对半拆分最长的笔划
    :param sketch:
    :param stk_len_max:
    :return:
    """
    strokes = sketch.copy()

    while True:
        max_stroke = max(strokes, key=lambda _stroke: du.stroke_length(_stroke))

        if du.stroke_length(max_stroke) <= stk_len_max:
            break
        else:
            du.single_split_(strokes)

    return strokes


def outlier_filter(stroke_list: list, len_thres, rect_dec_thres=0.3, near_rect_thres=0.1):
    """
    筛选出 outlier
    :param stroke_list:
    :param len_thres: 长度小于该值，可能为 outlier
    :param rect_dec_thres: 删除疑似 outlier 后，rect减小比例
    :param near_rect_thres: 删除疑似 outlier 后，rect减小比例
    :return:
    """
    # 计算草图 rect
    out_rect = du.get_rect(stroke_list)
    out_rect_area = out_rect.area()

    # 获取长度小于指定数值的笔划
    outliers = []
    valid_stks = []

    for c_stk in stroke_list:
        if du.stroke_length(c_stk) < len_thres and out_rect.is_near(c_stk, near_rect_thres):
            outliers.append(c_stk)
        else:
            valid_stks.append(c_stk)

    # 计算删除全部outlier后的草图的rect
    inner_rect_area = du.get_rect(valid_stks).area()

    if inner_rect_area / out_rect_area < 1.0 - rect_dec_thres:
        # 逐个将outliers内的元素添加进valid_stks，计算有效值
        for c_outlier in outliers:
            before_add_rect_area = du.get_rect(valid_stks).area()

            valid_stks.append(c_outlier)
            after_add_rect_area = du.get_rect(valid_stks).area()

            if after_add_rect_area / before_add_rect_area > 1 + rect_dec_thres:
                valid_stks.pop()

        return valid_stks
    else:
        return stroke_list






