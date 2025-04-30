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
        严格按照该距离采样，最后一个点向前插值到间隔距离
        :param dist:
        :return:
        """
        if dist >= self.arc_length:
            warnings.warn('resample dist is equal to stroke length, drop this stroke')
            return np.array([])

        else:
            interp_points = []
            c_arclen = 0.0

            while c_arclen < self.arc_length:
                interp_points.append(self.length_interp(c_arclen))
                c_arclen += dist

            # 向前插值最后一个点
            last_pnt = self.stk_points[-1]
            last_former_pnt = interp_points[-1]

            interp_dir = last_pnt - last_former_pnt
            norm = np.linalg.norm(interp_dir)

            if norm > 1e-5:
                interp_dir = interp_dir / norm
                interp_pnt = last_former_pnt + interp_dir * dist
                interp_points.append(interp_pnt)

            return np.vstack(interp_points)

    def length_interp(self, target_len):
        """
        返回从起点到该点处指定长度的点
        :param target_len:
        :return:
        """
        assert 0 <= target_len <= self.arc_length

        # 特殊情况，始末点
        if target_len < 1e-5:
            return self.stk_points[0]
        elif target_len > self.arc_length - 1e-5:
            return self.stk_points[-1]

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

    def batch_interp(self, paras) -> np.ndarray:
        interp_points = []
        for i in range(len(paras)):
            interp_points.append(self.single_interp(paras[i]))

        # # 目前只是点坐标，还需要加上每个点的属性
        interp_points = np.array(interp_points)
        # pen_attr = np.zeros_like(interp_points)
        # pen_attr[:, 0] = 17
        # pen_attr[-1, 0] = 16
        # interp_points = np.concatenate([interp_points, pen_attr], axis=1)

        return interp_points

    def single_interp(self, para: np.ndarray) -> np.ndarray:
        """
        参数为单位弧长参数
        :param para: np.ndarray [n, 2]
        :return:
        """
        assert 0 <= para <= 1

        if para < 1e-5:
            return self.stk_points[0]
        elif para > 1.0 - 1e-5:
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
        if direc_len < 1e-5:
            warnings.warn('left and right points are too close, return left point')
            return right_point

        else:
            direc /= direc_len
            target_point = left_point + rest_len * direc

            return target_point


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
            chordal_length.append(np.linalg.norm(c_stk[i, :] - c_stk[i + 1, :]))

    median = np.median(chordal_length)
    median = median * mid_ratio

    resampled = []
    for c_stk in stroke_list:
        lin_interp = LinearInterp(c_stk)
        resampled.append(lin_interp.uni_dist_interp(median))

    return resampled


def uni_arclength_resample_strict(stroke_list, resp_dist) -> list:
    """
    均匀布点，相邻点之间距离严格为 resp_dist，最后一个点向前插值到间隔距离
    :param stroke_list:
    :param resp_dist:
    :return:
    """
    assert isinstance(stroke_list, list)

    resampled = []
    for c_stk in stroke_list:
        lin_interp = LinearInterp(c_stk)

        c_resped_stk = lin_interp.uni_dist_interp_strict(resp_dist)

        if c_resped_stk.size != 0:
            resampled.append(c_resped_stk)

    return resampled


def uni_arclength_resample_strict_single(stroke, resp_dist) -> np.ndarray:
    """
    均匀布点，相邻点之间距离严格为 resp_dist，最后一个点向前插值到间隔距离
    :param stroke:
    :param resp_dist:
    :return:
    """

    lin_interp = LinearInterp(stroke)
    stroke_resampled = lin_interp.uni_dist_interp_strict(resp_dist)

    return stroke_resampled


def uni_arclength_resample_certain_pnts_single(stroke, n_point) -> np.ndarray:
    """
    均匀布点，单个笔划上的点数相同
    :param stroke:
    :param n_point:
    :return:
    """
    paras = np.linspace(0, 1, n_point).tolist()

    lin_interp = LinearInterp(stroke)
    stroke_resampled = lin_interp.batch_interp(paras)

    return stroke_resampled


def uni_arclength_resample_certain_pnts_batched(stroke_list, n_point) -> list:
    """
    使得草图中每个笔划上的点数相同
    :param stroke_list:
    :param n_point:
    :return:
    """
    resampled = []

    for c_stk in stroke_list:
        resampled.append(uni_arclength_resample_certain_pnts_single(c_stk, n_point))

    return resampled


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
