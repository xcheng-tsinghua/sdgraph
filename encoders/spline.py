"""
the parameter range of all BSpline curves are [0, 1]

type of stroke: ndarray[n_point, 2]
type of sketch: (list)[ndarray_1[n_point, 2], ndarray_2[n_point, 2], ..., ndarray_n[n_point, 2]]
"""

import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, BSpline, make_interp_spline, CubicSpline, interp1d
import numpy as np
import math
import warnings

import global_defs


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


class LinearInterp(object):
    def __init__(self, stk_points):
        self.stk_points = stk_points[:, :2]

        # 计算总弧长
        dist_to_previous = np.sqrt(np.sum(np.diff(self.stk_points, axis=0) ** 2, axis=1))
        self.cumulative_dist = np.concatenate(([0], np.cumsum(dist_to_previous)))

        self.arc_length = self.cumulative_dist[-1]

    def __call__(self, paras):
        """
        线性插值
        :param paras: 参数或参数列表
        :return: ndarray[n, 4]
        """
        if isinstance(paras, (float, int)):
            return self.single_interp(paras)

        else:
            return self.batch_interp(paras)

    def uni_dist_interp(self, dist):
        """
        尽量让点之间距离相等
        :param dist:
        :return:
        """
        if dist >= self.arc_length:
            warnings.warn('resample dist is equal to stroke length, drop this sketch')
            return None

        else:
            # 计算均分数，尽量大
            n_sections = math.ceil(self.arc_length / dist)
            paras = np.linspace(0, 1, n_sections + 1)

            return self.batch_interp(paras)

    def uni_dist_interp_strict(self, dist) -> np.ndarray:
        """
        严格按照该距离采样，可能丢失最后一个点
        :param dist:
        :return:
        """
        if dist >= self.arc_length:
            warnings.warn('resample dist is equal to stroke length, drop this stroke')
            return np.array([])

        else:
            interp_points = []
            c_arclen = 1e-5

            while c_arclen < self.arc_length:
                interp_points.append(self.length_interp(c_arclen))
                c_arclen += dist

            return np.vstack(interp_points)

    def length_interp(self, target_len):
        """
        返回从起点到该点处指定长度的点
        :param target_len:
        :return:
        """
        assert 0 <= target_len <= self.arc_length

        # cumulative[left_idx] <= target_len <= cumulative[left_idx + 1]
        left_idx = np.searchsorted(self.cumulative_dist, target_len) - 1

        # 在左右两点之间使用线性插值找到中间点
        rest_len = target_len - self.cumulative_dist[left_idx]

        left_point = self.stk_points[left_idx]
        right_point = self.stk_points[left_idx + 1]

        direc = right_point - left_point
        direc_len = np.linalg.norm(direc)

        # 左右点过于接近
        if direc_len < 1e-5:
            warnings.warn('left and right points are too close, return left point')
            return right_point

        else:
            direc /= direc_len
            target_point = left_point + rest_len * direc

            return target_point

    def batch_interp(self, paras):
        interp_points = []
        for i in range(len(paras)):
            interp_points.append(self.single_interp(paras[i]))

        # 目前只是点坐标，还需要加上每个点的属性
        interp_points = np.array(interp_points)
        pen_attr = np.zeros_like(interp_points)
        pen_attr[:, 0] = 17
        pen_attr[-1, 0] = 16
        interp_points = np.concatenate([interp_points, pen_attr], axis=1)

        return interp_points

    def single_interp(self, para):
        assert 0 <= para <= 1

        if para == 0:
            return self.stk_points[0]
        elif para == 1:
            return self.stk_points[-1]

        # 计算参数对应的弧长
        target_len = para * self.arc_length

        # cumulative[left_idx] <= target_len <= cumulative[left_idx + 1]
        left_idx = np.searchsorted(self.cumulative_dist, target_len) - 1

        # 在左右两点之间使用线性插值找到中间点
        rest_len = target_len - self.cumulative_dist[left_idx]

        left_point = self.stk_points[left_idx]
        right_point = self.stk_points[left_idx + 1]

        direc = right_point - left_point
        direc_len = np.linalg.norm(direc)

        # 左右点过于接近
        if direc_len < 1e-4:
            warnings.warn('left and right points are too close, return left point')
            return right_point

        else:
            direc /= direc_len
            target_point = left_point + rest_len * direc

            return target_point


def stroke_length(stroke):
    # stroke = stroke[:, :2]
    #
    # dist_all = 0.0
    #
    # for i in range(1, len(stroke)):
    #     dist_all += np.linalg.norm(stroke[i] - stroke[i - 1])
    #
    # return dist_all
    if stroke.shape[0] < 2:
        return 0.0

    stroke = stroke[:, :2]
    diffs = np.diff(stroke, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    return np.sum(segment_lengths)


def stroke_length_var(stroke_list):
    """
    计算草图中的笔划长度方差
    :param stroke_list:
    :return:
    """
    stroke_length_all = []

    for c_stk in stroke_list:
        stroke_length_all.append(stroke_length(c_stk))

    return np.var(stroke_length_all)


def bspline_knot(degree=5, n_control=6):
    """
    compute BSpline knot vector
    :param degree:
    :param n_control:
    :return:
    """
    n = n_control - 1
    p = degree
    m = n + p + 1

    n_knot_mid = n - p
    knot_vector = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n_knot_mid + 2),
        np.ones(degree)
    ))

    # 确保节点向量长度正确
    if len(knot_vector) != m + 1:
        raise ValueError(f'values in knot vector should equal to {m + 1}, but obtained {len(knot_vector)}')

    return knot_vector


def bspline_basis(t, degree, knots, n_pole):
    """
    compute BSpline basis function
    :param t: parameters of each points
    :param degree:
    :param knots:
    :param n_pole:
    :return:
    """
    basis = np.zeros((len(t), n_pole))
    for i in range(n_pole):
        coeff = np.zeros(n_pole)
        coeff[i] = 1
        spline = BSpline(knots, coeff, degree)
        basis[:, i] = spline(t)

    return basis


def chord_length_parameterize(points):
    """
    累积弦长参数化
    :param points: input points [n, 2]
    :return: parameters of each point
    """
    # 计算每个点到前一个点的距离
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))

    # np.cumsum 计算数组元素累积和, 除以总长度即为弦长参数，因为曲线参数在[0, 1]范围内
    # a = [x1, x2, ..., xn], np.cumsum(a) = [x1, x1 + x2, x1 + x2 + x3, ..., \sum_(k=1)^n(xk)]
    cumulative = np.concatenate(([0], np.cumsum(distances)))

    return cumulative / cumulative[-1]


def curve_length(spline, p_start, p_end, n_sec=100):
    """
    compute curva length
    :param spline:
    :param p_start: start parameter
    :param p_end: end parameter
    :param n_sec: compute sections
    :return: curve length between parameter range [p_start, p_end]
    """
    t_values = np.linspace(p_start, p_end, n_sec)
    # 获取曲线的导数
    derivative_spline = spline.derivative()
    # 在 t_values 上评估曲线导数
    derivatives = derivative_spline(t_values)
    # 计算弧长增量
    lengths = np.sqrt((derivatives[:, 0]) ** 2 + (derivatives[:, 1]) ** 2)
    # 使用梯形积分计算总弧长
    arc_length = np.trapz(lengths, t_values)

    return arc_length


def arclength_uniform(spline, num_points=100):
    """
    sample points from a BSpline curve follow curve length uniform
    :param spline: target BSpline
    :param num_points: number of points to be sampled
    :return: curve length between parameter range [p_start, p_end]
    """
    # 总参数范围
    t_min, t_max = 0., 1.
    # 参数值的细分，用于计算累积弧长
    t_values = np.linspace(t_min, t_max, 1000)
    arc_lengths = np.zeros_like(t_values)
    for i in range(1, len(t_values)):
        arc_lengths[i] = arc_lengths[i - 1] + curve_length(spline, t_values[i - 1], t_values[i], 30)

    # 总弧长
    total_arc_length = arc_lengths[-1]
    # 生成均匀分布的弧长
    uniform_arc_lengths = np.linspace(0, total_arc_length, num_points)
    # 根据弧长查找对应的参数值
    uniform_t_values = np.interp(uniform_arc_lengths, arc_lengths, t_values)
    # 使用参数值计算均匀采样点的坐标
    sampled_points = spline(uniform_t_values)

    return sampled_points, uniform_t_values


def bspline_approx(data_points, degree=3, n_pole=6, n_sample=100, sample_mode='e-arc', view_res=False):
    """
    给定一系列点拟合BSpline曲线，曲线严格通过首末点，但不一定通过中间点
    :param data_points: points to be approximated [n, 2]
    :param degree: 曲线次数
    :param n_pole: 控制点数
    :param n_sample: 返回的采样点数
    :param sample_mode: 重构曲线上的采样方法, 'e-arc': equal arc length sample, 'e-para': equal parameter sample
    :param view_res: is view approximated results
    :return: (sample points, fitting curve)
    """
    if data_points.shape[0] < degree + n_pole:
        raise ValueError('too less points in a stroke')

    # 1. 准备数据点
    x = data_points[:, 0]
    y = data_points[:, 1]

    # 2. 参数化数据点（弦长参数化）
    t = chord_length_parameterize(data_points)

    # 3. 定义B样条的节点向量
    knot_vector = bspline_knot(degree, n_pole)

    # 4. 构建基函数矩阵
    B = bspline_basis(t, degree, knot_vector, n_pole)

    # 5. 求解控制点的最小二乘问题
    ctrl_pts_x, _, _, _ = np.linalg.lstsq(B, x, rcond=None)
    ctrl_pts_y, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    control_points = np.vstack((ctrl_pts_x, ctrl_pts_y)).T

    # 6. 构建B样条曲线
    spline = BSpline(knot_vector, control_points, degree)

    if sample_mode == 'e-para':
        t_fine = np.linspace(0, 1, n_sample)
        curve_points = spline(t_fine)

    elif sample_mode == 'e-arc':
        curve_points, _ = arclength_uniform(spline, n_sample)

    else:
        raise ValueError('unknown sample mode.')

    if view_res:
        # 8. 可视化结果
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'ro', label='data point')
        plt.plot(control_points[:, 0], control_points[:, 1], 'k--o', label='ctrl point')
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='fitting bspline')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return curve_points, spline


def batched_spline_approx(point_list, min_sample=10, max_sample=100, approx_mode='linear-interp', degree=3, n_pole=6, sample_mode='e-arc', view_res=False, median_ratio=0.1) -> list:
    """

    :param point_list:
    :param approx_mode:
        'bspline': designed,
        'bspline-scipy': from scipy,
        'cubic-interp': cubic spline interp,
        'linear-interp': linear interp
        'uni-arclength': specify dist between points

    :param degree:
    :param n_pole:
    :param min_sample:
    :param max_sample:
    :param sample_mode:
    :param view_res:
    :param median_ratio: the point dist is (point dist median in a sketch) * median_ratio
    :return:
    """
    approx_list = []

    if approx_mode == 'uni-arclength':
        approx_list = uni_arclength_resample(point_list, median_ratio)

    else:
        for c_stroke in point_list:

            n_stkpnts = len(c_stroke)
            if n_stkpnts >= 2:

                if n_stkpnts <= min_sample:
                    n_sample = min_sample
                elif n_stkpnts >= max_sample:
                    n_sample = max_sample
                else:
                    n_sample = n_stkpnts

                if approx_mode == 'bspline':
                    approx_list.append(bspline_approx(c_stroke, degree, n_pole, n_sample, sample_mode, view_res)[0])
                elif approx_mode == 'bspline-scipy':
                    approx_list.append(bspl_approx2(c_stroke, n_sample, degree))
                elif approx_mode == 'cubic-interp':
                    approx_list.append(cubic_spline_resample(c_stroke, n_sample))
                elif approx_mode == 'linear-interp':
                    approx_list.append(linear_resample(c_stroke, n_sample))
                else:
                    ValueError('error approx mode')
            else:
                ValueError('points in stroke is lower than 2')

    # 删除数组中无效的None笔划
    approx_list = [x for x in approx_list if x is not None]

    return approx_list


def pnt_dist(p1, p2):
    """
    计算两点之间的欧氏距离
    """
    return np.linalg.norm(p1 - p2)


def uni_arclength_resample(stroke_list, mid_ratio=0.1):
    """
    将笔划均匀布点，使得点之间的距离尽量相同，使用线性插值
    :param stroke_list:
    :param mid_ratio: 弦长中位数比例
    :return:
    """
    # 计算弦长中位数
    chordal_length = []
    for c_stk in stroke_list:
        for i in range(c_stk.shape[0] - 1):
            chordal_length.append(pnt_dist(c_stk[i, :], c_stk[i + 1, :]))

    median = np.median(chordal_length)
    median = median * mid_ratio

    resampled = []
    for c_stk in stroke_list:
        lin_interp = LinearInterp(c_stk)
        resampled.append(lin_interp.uni_dist_interp(median))

    return resampled


def uni_arclength_resample_strict(stroke_list, resp_dist) -> list:
    """
    均匀布点，相邻点之间距离严格为 resp_dist，可能笔划中丢失最后一个点
    :param stroke_list:
    :param resp_dist:
    :return:
    """
    resampled = []
    for c_stk in stroke_list:
        lin_interp = LinearInterp(c_stk)

        c_resped_stk = lin_interp.uni_dist_interp_strict(resp_dist)

        if c_resped_stk.size != 0:
            resampled.append(c_resped_stk)

    return resampled


def uni_arclength_resample_strict_single(stroke, resp_dist) -> np.ndarray:
    """
    均匀布点，相邻点之间距离严格为 resp_dist，可能笔划中丢失最后一个点
    :param stroke:
    :param resp_dist:
    :return:
    """

    lin_interp = LinearInterp(stroke)
    stroke_resampled = lin_interp.uni_dist_interp_strict(resp_dist)

    return stroke_resampled


def bspl_approx2(points, n_samples=100, degree=3):
    x, y = points[:, 0], points[:, 1]

    plt.clf()
    plt.plot(x, y)
    plt.show()

    tck, u = splprep([x, y], k=degree, s=5)  # k=3为三次样条，s为平滑因子
    fit_x, fit_y = splev(np.linspace(0, 1, n_samples), tck)

    plt.clf()
    plt.plot(fit_x, fit_y)
    plt.show()

    curve_points = np.hstack((fit_x, fit_y))
    return curve_points


def cubic_spline_resample(points, n_samples=100):
    """
    使用三次样条插值方法，将二维点插值为曲线，并均匀取k个点。

    Parameters:
        points (numpy.ndarray): 二维点数组，大小为[n, 2]
        n_samples (int): 需要在曲线上取的点数

    Returns:
        sampled_points (numpy.ndarray): 在曲线上均匀分布的k个点，大小为[k, 2]
    """
    # 提取x和y坐标
    x = points[:, 0]
    y = points[:, 1]

    plt.clf()
    plt.plot(x, y)
    plt.show()

    # 计算累积弧长，用于生成参数t
    t = chord_length_parameterize(points)

    # 构建三次样条插值函数
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)

    # 在参数t范围内均匀取k个点
    t_uniform = np.linspace(0, 1, n_samples)

    # 计算均匀取点的坐标
    sampled_x = spline_x(t_uniform)
    sampled_y = spline_y(t_uniform)
    sampled_points = np.column_stack((sampled_x, sampled_y))

    plt.clf()
    plt.plot(sampled_x, sampled_y)
    plt.show()

    return sampled_points


def linear_resample(points, n_sample, sample_mode='arc'):
    """
    使用线性插值方法，将二维点插值为曲线，并均匀取k个点。

    Parameters:
        points (numpy.ndarray): 二维点数组，大小为[n, 2]
        n_sample (int): 需要在曲线上取的点数
        sample_mode (str): 'para': equal param sample; 'chordal': equal chordal error sample; 'arc': equal arc length

    Returns:
        sampled_points (numpy.ndarray): 在曲线上均匀分布的k个点，大小为[k, 2]
    """
    interp_curve = LinearInterp(points)
    paras = np.linspace(0, 1, n_sample)
    interp_pnts = interp_curve(paras)

    # show_pnts_with_idx(interp_pnts)

    return interp_pnts


    #
    # # 提取x和y坐标
    # x = points[:, 0]
    # y = points[:, 1]
    #
    # # 计算累积弧长，用于生成参数t
    # t = chord_length_parameterize(points)
    #
    # # 构建线性插值函数
    # linear_interp_x = interp1d(t, x, kind='linear')
    # linear_interp_y = interp1d(t, y, kind='linear')
    #
    # if sample_mode == 'para':
    #     # 在参数t范围内均匀取k个点
    #     t_uniform = np.linspace(0, 1, n_sample)
    #
    # elif sample_mode == 'chordal':
    #     # 在原始曲线上均匀插值较多点，用于计算曲率
    #     t_dense = np.linspace(0, 1, 1000)
    #     dense_x = linear_interp_x(t_dense)
    #     dense_y = linear_interp_y(t_dense)
    #
    #     # 计算每段曲率（近似）
    #     dx = np.gradient(dense_x)
    #     dy = np.gradient(dense_y)
    #     ddx = np.gradient(dx)
    #     ddy = np.gradient(dy)
    #
    #     curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    #     curvature = np.nan_to_num(curvature, nan=0.0)  # 处理除零的情况
    #
    #     # 计算累积权重，基于曲率调整分布
    #     weights = 1 + curvature  # 权重与曲率正相关
    #     cumulative_weights = np.cumsum(weights)
    #     cumulative_weights /= cumulative_weights[-1]
    #
    #     # 按累积权重均匀采样k个点
    #     t_uniform = np.interp(np.linspace(0, 1, n_sample), cumulative_weights, t_dense)
    #
    # else:
    #     raise ValueError('unknown sample mode')
    #
    # # 计算均匀取点的坐标
    # sampled_x = linear_interp_x(t_uniform)
    # sampled_y = linear_interp_y(t_uniform)
    # sampled_points = np.column_stack((sampled_x, sampled_y))
    #
    # return sampled_points


def show_pnts_with_idx(points):

    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points')

    # 添加索引标注
    for i, (x, y) in enumerate(points):
        plt.text(x, y, str(i), fontsize=12, color='red', ha='right', va='bottom')

    # 添加标题和坐标轴标签
    plt.title("Scatter Plot with Point Indices")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # 显示图像
    plt.grid(True)
    plt.show()


def stk_pnt_num_filter(sketch, min_point=5) -> list:
    """
    filter out strokes whose point number is below min_point
    :param sketch: (list) element: ndarray[n_point, 2]
    :param min_point: (int)
    :return:
    """
    # 如果是ndarray，说明笔划未分割
    if isinstance(sketch, np.ndarray):
        sketch = sketch_split(sketch)

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

    # 如果是ndarray，说明笔划未分割，直接以未分割的点进行处理
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
        stroke_list = sketch_split(stroke_list)

    filtered_stk = []

    for c_stk in stroke_list:
        c_stroke_length = stroke_length(c_stk)
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
        sketch = sketch_split(sketch)

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
        sketch = sketch_split(sketch)

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
        sketch = sketch_split(sketch)

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


def single_split_(stroke_list: list):
    """
    将草图中最长的笔画对半分割成两个
    :param stroke_list:
    :return:
    """
    # stroke_list = copy.deepcopy(stk_list)

    # if len(stroke_list) == 0:
    #     asasas = 0

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


def stk_num_minimal_filter(sketch, n_stk_min):
    """
    保证草图中的笔划数不能小于指定值，否则递归分割最长笔划，指导符合为止
    :param sketch:
    :param n_stk_min:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        sketch = sketch_split(sketch)

    while True:
        if len(sketch) >= n_stk_min:
            break

        single_split_(sketch)

    return sketch


def valid_stk_filter(sketch: list):
    """
    某些笔划可能为空，即该位置的笔划为size=(0,)，需要删除
    :param sketch:
    :return:
    """
    sketch_new = []

    for c_stk in sketch:
        if c_stk.size != 0:
            sketch_new.append(c_stk)

    return sketch_new


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
            single_split_(strokes)

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


if __name__ == '__main__':
    # bspline_interp1(4, 6)

    # datas = np.array([
    #     [0, 0],
    #     [1, 2],
    #     [2, 3],
    #     [4, 3],
    #     [5, 2],
    #     [6, 0],
    #     [8, 3.2],
    #     [9, 6.5],
    #     [8, -5],
    # ])
    #
    # bspline_approx(data_points=datas, view_res=True, n_sample=10, n_pole=6, degree=2, sample_mode='e-para')

    sketch_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Key\0a4b71aa11ae34effcdc8e78292671a3_3.txt'

    new_sketch = sketch_split(np.loadtxt(sketch_root, delimiter=','))
    new_sketch = near_pnt_dist_filter(new_sketch, 0.001)
    new_sketch = stk_pnt_double_filter(new_sketch)

    # for c_stk_ in new_sketch:
    #     plt.scatter(c_stk_[:, 0], -c_stk_[:, 1])
    #     print(f'当前笔划中点数：{len(c_stk_)}')
    # plt.show()

    pass
