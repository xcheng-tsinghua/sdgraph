import numpy
import numpy as np

from data_utils import sketch_utils
from data_utils import vis
from data_utils import preprocess
from data_utils import sketch_utils as du
import random
from matplotlib import pyplot as plt

import global_defs


def traverse_folder():
    data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut'

    files_all = sketch_utils.get_allfiles(data_root)
    random.shuffle(files_all)

    for c_file in files_all:
        print(c_file)
        vis.vis_sketch_orig(c_file, title=c_file, show_dot=True, dot_gap=1)
        preprocess.preprocess_orig(c_file, is_show_status=True)
        # sketch_utils.std_to_tensor_img(np.loadtxt(c_file, delimiter=','))


def draw_main_fig():
    the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e667cbb1491b6cf657d8627e60604c7c_3.txt'
    # vis.vis_sketch_orig(the_file, show_dot=True, dot_gap=3)

    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = du.load_sketch_file(the_file, delimiter=',')

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

    strokes_filter = []

    for idx, s in enumerate(strokes):
        if idx == 2:
            s = s[::7, :]

        if du.stroke_length(s) > 0.3:
            strokes_filter.append(s)





    dot_gap = 1
    for s in strokes_filter:

        plt.clf()

        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])
        plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

        plt.axis('off')
        plt.axis("equal")
        plt.show()

    # plt.axis('off')
    # plt.axis("equal")
    # plt.show()





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

    draw_main_fig()


    pass












