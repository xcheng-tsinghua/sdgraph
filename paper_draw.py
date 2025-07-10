import numpy
import numpy as np

from data_utils import sketch_utils
from data_utils import vis
from data_utils import preprocess as prep
from data_utils import sketch_utils as du
import random
from matplotlib import pyplot as plt
from encoders import spline as sp
from tqdm import tqdm
from data_utils import sketch_file_read as fr

import global_defs


def vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, title=None, show_dot=False, show_axis=False, dot_gap=1, save_idx=0):
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
    sketch_data = fr.load_sketch_file(root, delimiter=',')

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
    plt.figure(figsize=(10, 5))
    for s in strokes:
        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])

        if show_dot:
            plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

    if not show_axis:
        plt.axis('off')

    plt.axis("equal")
    plt.title(title)
    plt.savefig(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\plug-test\{save_idx}.png')
    plt.clf()
    plt.close()
    # plt.show()



def traverse_folder(data_root=r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Plug'):
    # data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut'

    files_all = sketch_utils.get_allfiles(data_root)
    random.shuffle(files_all)

    for idx, c_file in tqdm(enumerate(files_all), total=len(files_all)):
        # print(c_file)
        vis_sketch_orig(c_file, title=c_file, show_dot=True, dot_gap=1, save_idx=idx)
        # prep.preprocess_orig(c_file, is_show_status=True)
        # sketch_utils.std_to_tensor_img(np.loadtxt(c_file, delimiter=','))


def draw_main_fig():
    def expand_array(arr):
        k = arr.shape[0]  # 获取数组的行数

        # 创建一个形状为(k, 1)的全1数组
        new_column = np.ones((k, 1), dtype=arr.dtype)

        # 将最后一个元素设为0
        new_column[-1, 0] = 0

        # 水平拼接原数组和新列
        return np.hstack((arr, new_column))

    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e667cbb1491b6cf657d8627e60604c7c_3.txt'
    the_file = r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source.txt'
    # vis.vis_sketch_orig(the_file, show_dot=True, dot_gap=3)

    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # -------------------------------
    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 重采样，使得点之间的距离近似相等
    # strokes = sp.batched_spline_approx(
    #     point_list=strokes,
    #     median_ratio=0.1,
    #     approx_mode='uni-arclength'
    # )

    # for idx, s in enumerate(strokes):
    #     if idx == 2:
    #         fs = s[::5, :]
    #         fs[-1, :] = s[-1, :]
    #         s = fs
    #
    #     # if idx == 12:
    #     #     s = s[:13, :]
    #
    #     if du.stroke_length(s) > 0.3:
    #         strokes_filter.append(s)

    # np.savetxt(r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source.txt', np.vstack(strokes_filter), delimiter=',')

    sf = []
    for idx, s in enumerate(strokes):
        if idx == 0:
            s = s[:-4, :]

        if idx == 1:
            s = s[:-3, :]

        if idx == 2:
            s = s[:-1, :]

        if idx == 3:
            s = s[7:-3, :]

        if idx == 4:
            s = s[5:, :]

        if idx == 5:
            s = s[:-3, :]

        if idx == 6:
            s = s[:-2, :]

        if idx == 7:
            s = s[4:, :]

        if idx == 8:
            s = s[4:-3, :]

        if idx == 9:
            s = s[3:-2, :]

        if idx == 10:
            s = s[:-2, :]

        if idx == 11:
            s = s[:-6, :]

        if idx == 15:
            s = s[:-3, :]

        sf.append(s[:, :2])

    strokes_merge = [sf[0], sf[1], sf[2]]
    strokes_merge.append(np.vstack([np.flip(sf[4], axis=0), sf[3], sf[6], sf[5], np.flip(sf[7], axis=0), np.flip(sf[8], axis=0)]))
    # strokes_merge.append(np.vstack([sf[9], sf[13], np.flip(sf[10], axis=0)]))
    strokes_merge.append(sf[9])
    strokes_merge.append(np.flip(sf[10], axis=0))
    strokes_merge.append(np.vstack([sf[12], sf[15], sf[13], sf[14], np.flip(sf[11], axis=0)]))

    plt.figure(figsize=(10, 5))

    dot_gap = 2

    altered_stks_save = []

    for idx, s in enumerate(strokes_merge):
        altered_stks_save.append(expand_array(s))

        # plt.clf()

        # np.savetxt(fr'C:\Users\ChengXi\Desktop\fig\fig1\nut_stk_{idx}.txt', s, delimiter=',')

        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

        # plt.title(f'stk index: {idx}')
        # plt.axis('off')
        # plt.axis("equal")
        # plt.show()

    # for idx, s in enumerate(strokes_filter):
    #     plt.scatter([s[0, 0]], [-s[0, 1]], s=80, c='black')

    np.savetxt(r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_merged2.txt', np.vstack(altered_stks_save), delimiter=',')

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def show_fig1():
    the_file = r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_merged2.txt'
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为0，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    strokes = sp.uni_arclength_resample_strict(strokes, 0.1)

    plt.figure(figsize=(10, 5))

    colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189), (140,86,75), (227,119,194)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    dot_gap = 1

    show_fig_idx = 6

    def replace_except_index_inplace(arr, index):
        for i in range(len(arr)):
            if i != index:
                arr[i] = (1, 1, 1)
        return arr

    poins_resamp = []

    for idx, s in enumerate(strokes):

        if idx == 0:
            s = s[:-1, :]

        if idx == 1:
            s = s[:-1, :]

        if idx == 3:
            s = s[:-1, :]

        if idx == 6:
            s = s[:-1, :]

        poins_resamp.append(s)

        replace_except_index_inplace(colors, show_fig_idx)


        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=colors[idx])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=colors[idx])

    plt.plot(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], color=colors[show_fig_idx])
    plt.scatter(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], s=80, color=colors[show_fig_idx])

    np.savetxt(r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_points.txt', np.vstack(poins_resamp), delimiter=',')

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def show_points():
    the_file = r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_points.txt'
    points = np.loadtxt(the_file, delimiter=',')

    point_skip = 4
    plt.figure(figsize=(10, 5))
    plt.scatter(points[::point_skip, 0], -points[::point_skip, 1], s=80)

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def show_fig2():
    the_file = r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_merged2.txt'
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为0，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    strokes = sp.uni_arclength_resample_strict(strokes, 0.1)

    plt.figure(figsize=(10, 5))

    # colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189), (140,86,75), (227,119,194)]
    colors = [(31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180),
              (31,119,180)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    dot_gap = 2

    show_fig_idx = 6

    def replace_except_index_inplace(arr, index):
        for i in range(len(arr)):
            if i != index:
                arr[i] = (1, 1, 1)
        return arr

    poins_resamp = []

    for idx, s in enumerate(strokes):

        if idx == 0:
            s = s[:-1, :]

        if idx == 1:
            s = s[:-1, :]

        if idx == 3:
            s = s[:-1, :]

        if idx == 6:
            s = s[:-1, :]

        poins_resamp.append(s)

        replace_except_index_inplace(colors, show_fig_idx)


        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=colors[idx])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=colors[idx])

    # plt.plot(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], color=colors[show_fig_idx])
    plt.scatter(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], s=80, color=colors[show_fig_idx])

    np.savetxt(r'C:\Users\ChengXi\Desktop\fig\fig1\nut_source_points.txt', np.vstack(poins_resamp), delimiter=',')

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def show_fig3():
    the_file = r'C:\Users\ChengXi\Desktop\fig\vec_sketch\nut_source_merged2.txt'
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为0，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    strokes = sp.uni_arclength_resample_strict(strokes, 0.1)

    plt.figure(figsize=(10, 5))

    # colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189), (140,86,75), (227,119,194)]
    colors = [(31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180),
              (31,119,180)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    dot_gap = 5

    poins_resamp = []

    for idx, s in enumerate(strokes):

        if idx == 0:
            s = s[:-1, :]

        if idx == 1:
            s = s[:-1, :]

        if idx == 3:
            s = s[:-1, :]

        if idx == 6:
            s = s[:-1, :]

        poins_resamp.append(s)

        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=colors[idx])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=colors[idx])

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def add_noise():
    the_file = r'C:\Users\ChengXi\Desktop\fig\vec_sketch\nut_source_merged2.txt'
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    # 添加噪音
    np.random.seed(99)
    random_array1 = np.random.randn(*coordinates.shape)
    random_array2 = np.random.randn(*coordinates.shape)
    # coordinates = 0.9 * coordinates + 0.05 * random_array1 + 0.05 * random_array2
    coordinates = 0.96 * coordinates + 0.04 * random_array1

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为0，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    strokes = sp.uni_arclength_resample_strict(strokes, 0.1)

    plt.figure(figsize=(10, 5))

    colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189), (140,86,75), (227,119,194)]
    # colors = [(31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    dot_gap = 1

    poins_resamp = []

    show_fig_idx = 7
    for idx, s in enumerate(strokes):

        if idx == 0:
            s = s[:-1, :]

        if idx == 1:
            s = s[:-1, :]

        if idx == 3:
            s = s[:-1, :]

        if idx == 6:
            s = s[:-1, :]

        poins_resamp.append(s)

        if idx == show_fig_idx:
            sel_stk = s

        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=colors[idx])
        # plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=colors[idx])

        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=[1,1,1])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=[1,1,1])

    plt.plot(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], color=colors[show_fig_idx])
    plt.scatter(strokes[show_fig_idx][::dot_gap, 0], -strokes[show_fig_idx][::dot_gap, 1], s=80, color=colors[show_fig_idx])

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def add_noise2():
    the_file = r'C:\Users\ChengXi\Desktop\fig\vec_sketch\nut_source_merged2.txt'
    sketch_data = fr.load_sketch_file(the_file, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    # 添加噪音
    np.random.seed(99)
    random_array1 = np.random.randn(*coordinates.shape)
    random_array2 = np.random.randn(*coordinates.shape)
    # coordinates = 0.9 * coordinates + 0.05 * random_array1 + 0.05 * random_array2
    coordinates = 0.96 * coordinates + 0.04 * random_array1
    # coordinates = 0.04 * random_array1

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为0，防止出现空数组
    sketch_data[-1, 2] = global_defs.pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    strokes = sp.uni_arclength_resample_strict(strokes, 0.1)

    plt.figure(figsize=(10, 5))

    # colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40), (148,103,189), (140,86,75), (227,119,194)]
    colors = [(31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180), (31,119,180)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    dot_gap = 1

    poins_resamp = []

    for idx, s in enumerate(strokes):

        if idx == 0:
            s = s[:-1, :]

        if idx == 1:
            s = s[:-1, :]

        if idx == 3:
            s = s[:-1, :]

        if idx == 6:
            s = s[:-1, :]

        poins_resamp.append(s)

        # plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1], color=colors[idx])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=colors[idx])
        # plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80, color=(31/255,119/255,180/255))

    plt.axis('off')
    plt.axis("equal")
    plt.show()


def plot_ordered_points(points):
    """
    绘制按顺序连接的点并在每个点附近显示行索引

    参数:
    points -- 形状为(n,2)的numpy数组，表示平面上的n个点
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制连接线（蓝色实线）
    ax.plot(points[:, 0], points[:, 1], 'b-o', linewidth=2, markersize=8,
            markerfacecolor='red', markeredgecolor='black')

    # 添加每个点的行索引标签
    for i, (x, y) in enumerate(points):
        # 计算偏移量（避免文本重叠在点上）
        offset_x = 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        offset_y = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        # 添加带背景框的文本标签
        ax.text(x + offset_x, y + offset_y, str(i),
                fontsize=12, weight='bold',
                bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round,pad=0.3'))

    # 设置图形属性
    ax.set_title("Ordered Points Connection", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # 自动调整坐标轴范围（增加10%的边界留白）
    x_padding = 0.1 * (np.max(points[:, 0]) - np.min(points[:, 0]))
    y_padding = 0.1 * (np.max(points[:, 1]) - np.min(points[:, 1]))
    ax.set_xlim(np.min(points[:, 0]) - x_padding, np.max(points[:, 0]) + x_padding)
    ax.set_ylim(np.min(points[:, 1]) - y_padding, np.max(points[:, 1]) + y_padding)

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # # data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all'
    # asasad = r'E:\document\iDesignCAD-Assistant\iDesignCAD\data.txt'
    # vis.vis_sketch_orig(asasad, title=asasad)

    # D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\3130b3f7df395748c6dc15f2cd637cb5_5.txt
    # D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\a4d57653396146040d9e5b45149d2ba1_2.txt

    # fig_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\a4d57653396146040d9e5b45149d2ba1_2.txt'
    #
    # vis.vis_sketch_orig(fig_file, show_dot=True, dot_gap=1)
    # preprocess.resample_stake(fig_file, is_show_status=True)

    # sketch_processed = preprocess.preprocess_outlier_resamp_seg(fig_file)
    # vis.vis_sketch_list(sketch_processed, show_dot=True)
    # show_points()

    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e0aa70a1d95a7e426cc6522eeddaa713_3.txt'  # 有效齿轮
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Plug\8d3fe8a25838031b3321b480176926db_3.txt'
    # vis.vis_sketch_orig(the_file, show_dot=True, dot_gap=3)

    # add_noise2()

    # prep.find_nonstandard_leaf_dirs(rf'/opt/data/private/data_set/quickdraw/mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # traverse_folder()

    # draw_main_fig()
    # show_fig1()
    show_fig3()

    # npz_file = r'D:\document\DeepLearning\DataSet\quickdraw\raw\airplane.full.npz'
    # c_sketch = fr.npz_read(npz_file)[0][0]
    # sketch_resample = prep.preprocess_orig(c_sketch)
    # sketch_resample = sketch_resample.reshape([-1, 2])
    # plot_ordered_points(sketch_resample)

    # svg_file_ = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketches_svg\airplane\n02691156_58-1.svg'
    # svg_transed = fr.svg_read(svg_file_)
    # vis.vis_sketch(svg_transed)


    pass












