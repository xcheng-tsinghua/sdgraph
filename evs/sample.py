import numpy as np


def upsample_keep_dense_single(stroke, n_sample):
    # 步骤1: 计算每一段的欧氏距离
    deltas = np.diff(stroke, axis=0)
    dists = np.linalg.norm(deltas, axis=1)

    # 步骤2: 累计长度（包括起点0）
    cumdist = np.concatenate([[0], np.cumsum(dists)])

    # 步骤3: 在累计长度上生成等距的 n_sample 个位置
    total_length = cumdist[-1]
    uniform_dist = np.linspace(0, total_length, n_sample)

    # 步骤4: 在原始路径上用线性插值插出对应的新点
    x = np.interp(uniform_dist, cumdist, stroke[:, 0])
    y = np.interp(uniform_dist, cumdist, stroke[:, 1])

    return np.stack([x, y], axis=1)


def upsample_keep_dense(stroke_list, n_sample):
    """
    每个笔划上采样到指定倍数的点
    :param stroke_list: 笔划列表
    :param n_sample: 上采样倍数
    :return:
    """




