import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import global_defs
from data_utils.sketch_utils import get_allfiles, get_subdirs

import encoders.spline as sp


def vis_sketch_folder(root):
    classes = get_subdirs(root)
    for c_class in classes:
        c_dir = os.path.join(root, c_class)
        c_files = get_allfiles(c_dir)

        for idx in range(3):
            c_file_show = c_files[idx]
            print(c_file_show)
            vis_sketch_orig(c_file_show)


def vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, title=None, show_dot=False, show_axis=False, dot_gap=1):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param root:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param title: 落笔指令
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

    # -------------------------------
    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    # 重采样，使得点之间的距离近似相等
    # strokes = sp.batched_spline_approx(
    #     point_list=strokes,
    #     median_ratio=0.1,
    #     approx_mode='uni-arclength'
    # )

    for s in strokes:
        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])

        if show_dot:
            plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

    if not show_axis:
        plt.axis('off')

    plt.axis("equal")
    plt.title(title)
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
    coordinates = coordinates.reshape([n_stroke, n_stk_pnt, 2])

    for i in range(n_stroke):
        plt.plot(coordinates[i, :, 0], -coordinates[i, :, 1])

        if show_dot:
            plt.scatter(coordinates[i, :, 0], -coordinates[i, :, 1])

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


def vis_sketch_list(strokes, show_dot=False, title=None):
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    # plt.axis('off')
    plt.axis("equal")
    plt.title(title)
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


def save_format_sketch_test(sketch_points, file_path, z_thres=0.5):
    """
    保存设定格式的草图
    :param sketch_points: [2, n_stk * n_stk_pnt]
    :param file_path:
    :param z_thres: z 位置大于该值才判定为有效笔划
    :return:
    """

    n_stk = sketch_points.size(0)

    # -> [n_stk, n_stk_pnt, channel]
    sketch_points = sketch_points.detach().cpu().numpy()

    plt.clf()
    for stk_idx in range(n_stk):
        c_stk = sketch_points[stk_idx]  # -> [n_stk_pnt, channel]

        # 去掉无效点
        c_stk = c_stk[c_stk[:, 2] >= z_thres]

        plt.plot(c_stk[:, 0], -c_stk[:, 1])

    plt.axis('off')
    plt.savefig(file_path)


def vis_false_log(log_root: str) -> None:
    # 读取每行
    with open(log_root, 'r') as f:
        for c_line in f.readlines():
            c_line = c_line.strip()
            c_file_show = c_line.replace('/opt/data/private/data_set', 'D:/document/DeepLearning/DataSet')
            print(c_line.split('/')[-2])
            print(c_file_show)
            vis_sketch_unified(c_file_show)


def test_vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, show_dot=False, show_axis=False):
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

    # -------------------------------
    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    # 重采样，使得点之间的距离近似相等
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        median_ratio=0.1,
        approx_mode='uni-arclength'
    )

    colors = [[31/255,119/255,180/255], [255/255,127/255,14/255], [44/255,160/255,44/255], [214/255,39/255,40/255], [148/255,103/255,189/255], [140/255,86/255,75/255], [227/255,119/255,194/255]]

    for s, color in zip(strokes, colors):
        # s = s[::105]  # 45
        plt.plot(s[:, 0], -s[:, 1], color=color)

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1], s=80, color=[31/255,119/255,180/255])

        # if not show_axis:
        #     plt.axis('off')
        # plt.show()

    if not show_axis:
        plt.axis('off')
    plt.show()


def test():
    from encoders.utils import index_points

    def get_coor(skh_root):
        # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
        sketch_data = np.loadtxt(skh_root, delimiter=',')

        # 2D coordinates
        coordinates = sketch_data[:, :2]

        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        return coordinates

    sketch1 = r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\16.txt'
    sketch2 = r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\16.txt'

    sketch1 = torch.from_numpy(get_coor(sketch1))  # [n, 2]
    sketch2 = torch.from_numpy(get_coor(sketch2))  # [n, 2]

    sketchs = torch.cat([sketch1.unsqueeze(0), sketch2.unsqueeze(0)], dim=0)  # [bs, n, 2]
    sketchs = sketchs.permute(0, 2, 1).contiguous()  # [bs, 2, n]
    sketchs = sketchs.view(2, 2, global_defs.n_stk, global_defs.n_stk_pnt)
    sketchs = sketchs.permute(0, 2, 3, 1).contiguous()  # [bs, n_stk, n_stk_pnt, 2]
    sketchs = sketchs.view(2, global_defs.n_stk, global_defs.n_stk_pnt * 2)  # [bs, n_stk, n_stk_pnt * 2]

    idx = torch.randint(0, global_defs.n_stk, [2, global_defs.n_stk // 2])
    sketchs = index_points(sketchs, idx)  # [bs, n_stk // 2, n_stk_pnt * 2]
    sketchs = sketchs.view(2, global_defs.n_stk // 2, global_defs.n_stk_pnt, 2)
    sketchs = sketchs[0, :, :, :]

    for i in range(global_defs.n_stk // 2):
        c_stk = sketchs[i, :, :]
        plt.plot(c_stk[:, 0], -c_stk[:, 1])
        # plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Gear\8646fb6b0a7f42bb9d5036995471b6b0_1.txt', show_dot=True)

    # show_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32_no_mix_proc\110.txt', show_dot=True)

    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\toothbrush\18246.txt')
    vis_sketch_orig(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt_cls\train\alarm clock\85.txt', show_dot=True)
    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\16.txt')

    # ahead, ext = os.path.splitext(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\train\apple\177.txt')

    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk32_stkpnt32\train\Gear\0dd1520e215d8d4c3cdbfe889316ba33_4.txt')

    # --- vis
    # vis_sketch_folder(r'D:\document\DeepLearning\DataSet\sketch_cad\sketch_txt\train')

    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\1.txt', show_dot=False)

    # test()

    # vis_sketch_unified(r'')

    # vis_false_log(r'C:\Users\ChengXi\Downloads\false_instance.txt')

    vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Bearing\f973078416a6819866b86970c22ae8f9_4.txt', show_dot=True)

    pass


