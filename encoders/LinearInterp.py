"""
linear interpolation of a list ordered points.
arc length parameter
points save in ndarray, such as [n, 2]
"""

import numpy as np


class LinearInterp(object):
    def __init__(self, stk_points):
        self.stk_points = stk_points

        # 计算总弧长
        dist_to_previous = np.sqrt(np.sum(np.diff(stk_points, axis=0) ** 2, axis=1))
        self.cumulative_dist = np.concatenate(([0], np.cumsum(dist_to_previous)))

        self.arc_length = self.cumulative_dist[-1]

    def __call__(self, paras):
        if isinstance(paras, (float, int)):
            return self.single_interp(paras)

        interp_points = []
        for i in range(len(paras)):
            interp_points.append(self.single_interp(paras[i]))

        return np.array(interp_points)

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
        direc /= np.linalg.norm(direc)

        target_point = left_point + rest_len * direc
        return target_point


if __name__ == '__main__':
    aasas = LinearInterp(None)

    aasas(0)

