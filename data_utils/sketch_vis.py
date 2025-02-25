import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import global_defs


def vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, show_dot=False, show_axis=False):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param root:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param show_dot:
    :param show_axis:
    :return:
    """
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = np.loadtxt(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    if not show_axis:
        plt.axis('off')
    plt.show()


def vis_sketch_data(sketch_data, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_scale=True, show_dot=False):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param sketch_data:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param is_scale: 是否将质心平移到 (0, 0)，且将草图大小缩放到 [-1, 1]^2
    :param show_dot:
    :return:
    """
    # 2D coordinates
    coordinates = sketch_data[:, :2]

    if is_scale:
        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.show()


def vis_sketch_unified(root, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False):
    """
    显示笔划与笔划点归一化后的草图
    """
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = np.loadtxt(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    coordinates = torch.from_numpy(coordinates)
    coordinates = coordinates.view(n_stroke, n_stk_pnt, 2)

    for i in range(n_stroke):
        plt.plot(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

        if show_dot:
            plt.scatter(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

    # plt.axis('off')
    plt.show()


def vis_unified_sketch_data(sketch_data, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False):
    """
    显示笔划与笔划点归一化后的草图
    """
    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    coordinates = torch.from_numpy(coordinates)
    coordinates = coordinates.view(n_stroke, n_stk_pnt, 2)

    for i in range(n_stroke):
        plt.plot(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

        if show_dot:
            plt.scatter(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

    # plt.axis('off')
    plt.show()


def vis_sketch_list(strokes, show_dot=False):
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.show()


def save_format_sketch(sketch_points, file_path, is_smooth=False):
    """
    保存设定格式的草图
    :param sketch_points: [2, n_stk * n_stk_pnt]
    :param file_path:
    :param is_smooth: 是否保存光顺后的草图
    :return:
    """

    def curve_smooth(x, y):
        tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度
        new_u = np.linspace(0, 1, 100)
        new_x, new_y = splev(new_u, tck)
        return new_x, new_y

    n_stk = global_defs.n_stk
    n_stk_pnt = global_defs.n_stk_pnt

    sketch_points = sketch_points.view(2, n_stk, n_stk_pnt).detach().cpu().numpy()

    plt.clf()
    for stk_idx in range(n_stk):
        plt.plot(sketch_points[0, stk_idx, :], -sketch_points[1, stk_idx, :])
        # plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.savefig(file_path)

    if is_smooth:
        plt.clf()
        for stk_idx in range(n_stk):
            c_stk = sketch_points[:, stk_idx, :]
            fit_x, fit_y = curve_smooth(c_stk[0, :], c_stk[1, :])
            plt.plot(fit_x, -fit_y)
            # plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        ahead, ext = os.path.splitext(file_path)
        plt.savefig(ahead + 'smooth' + ext)


if __name__ == '__main__':
    # show_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch\train\Bearing\00b11be6f26c85ca85f84daf52626b36_2.txt')

    # show_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32_no_mix_proc\110.txt', show_dot=True)

    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch\sketch_txt\train\Bolt\0e697fb4c47314eeaf2dbf6108f69040_1.txt', show_axis=True, show_dot=True)
    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch\train\Bearing\0bc12e6b9e792b74da4f7819d0041c9b_1.txt')

    # ahead, ext = os.path.splitext(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\train\apple\177.txt')

    vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk32_stkpnt32\train\Gear\0dd1520e215d8d4c3cdbfe889316ba33_4.txt')

    pass


