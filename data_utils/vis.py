import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2
import random

import global_defs
from data_utils.sketch_utils import get_allfiles, get_subdirs
import data_utils.sketch_utils as du
import encoders.spline as sp
from data_utils import sketch_file_read as fr


def vis_sketch_folder(root=r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all', shuffle=True, show_dot=True, dot_gap=3):
    files_all = get_allfiles(root)
    if shuffle:
        random.shuffle(files_all)

    for c_file in files_all:
        print(c_file)
        vis_sketch_orig(c_file, show_dot=show_dot, dot_gap=dot_gap)

    # classes = get_subdirs(root)
    # for c_class in classes:
    #     c_dir = os.path.join(root, c_class)
    #     c_files = get_allfiles(c_dir)
    #
    #     for idx in range(3):
    #         c_file_show = c_files[idx]
    #         print(c_file_show)
    #         vis_sketch_orig(c_file_show, show_dot=show_dot, dot_gap=dot_gap)


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

    for s in strokes:
        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])

        if show_dot:
            plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

    if not show_axis:
        plt.axis('off')

    plt.axis("equal")
    plt.title(title)
    plt.show()


def vis_sketch_data(sketch_data, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, title=None, is_scale=False, show_dot=False):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param sketch_data:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param title:
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
    plt.title(title)
    plt.show()


def vis_s5_data(sketch_data, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down):
    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

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


def vis_unified_sketch_data(sketch_data, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False, title=None):
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
    plt.axis('equal')
    plt.title(title)
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


def save_format_sketch(sketch_points, file_path, is_smooth=False, is_near_merge=False):
    """
    保存设定格式的草图
    :param sketch_points: [n_stk, n_stk_pnt, 2]
    :param file_path:
    :param is_smooth: 是否保存光顺后的草图
    :param is_near_merge:
    :return:
    """
    def curve_smooth(x, y):
        tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度
        new_u = np.linspace(0, 1, 100)
        new_x, new_y = splev(new_u, tck)
        return new_x, new_y

    # 将过近的笔划合并
    n_stk, n_stk_pnt, channel = sketch_points.size()
    sketch_points = sketch_points.detach().cpu().numpy()

    stroke_list = []
    for i in range(n_stk):
        stroke_list.append(sketch_points[i])

    if is_near_merge:
        stroke_list = du.stroke_merge_until(stroke_list, 0.1)

    plt.clf()
    for stk_idx in range(len(stroke_list)):
        c_stk = stroke_list[stk_idx]
        plt.plot(c_stk[:, 0], -c_stk[:, 1])
        # plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.savefig(file_path)

    if is_smooth:
        plt.clf()
        for stk_idx in range(n_stk):
            c_stk = sketch_points[stk_idx, :, :]
            fit_x, fit_y = curve_smooth(c_stk[:, 0], c_stk[:, 1])
            plt.plot(fit_x, -fit_y)
            # plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        ahead, ext = os.path.splitext(file_path)
        plt.savefig(ahead + 'smooth' + ext)


def save_format_sketch_test(sketch_points, file_path, z_thres=0.0):
    """
    保存设定格式的草图
    :param sketch_points: [n_stk, n_stk_pnt, 3]
    :param file_path:
    :param z_thres: z 位置大于该值才判定为有效点
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


def vis_seg_imgs(npz_root=r'D:\document\DeepLearning\DataSet\sketch_seg\SketchSeg-150K'):
    def canvas_size(sketch, padding: int = 30):
        """
        :param sketch: n*3 or n*4
        :param padding: white padding, only make impact on visualize.
        :return: int list,[x, y, h, w], [startX, startY, canvasH, canvasY]
        """
        # get canvas size
        x_point = np.array([0])
        y_point = np.array([0])
        xmin_xmax = np.array([0, 0])
        ymin_ymax = np.array([0, 0])
        for stroke in sketch:
            delta_x = stroke[0]
            delta_y = stroke[1]
            if x_point + delta_x > xmin_xmax[1]:
                xmin_xmax[1] = x_point + delta_x
            elif x_point + delta_x < xmin_xmax[0]:
                xmin_xmax[0] = x_point + delta_x
            if y_point + delta_y > ymin_ymax[1]:
                ymin_ymax[1] = y_point + delta_y
            elif y_point + delta_y < ymin_ymax[0]:
                ymin_ymax[0] = y_point + delta_y
            x_point += delta_x
            y_point += delta_y

        # padding
        assert padding >= 0 and isinstance(padding, int)
        xmin_xmax += np.array([-padding, +padding])  # padding
        ymin_ymax += np.array([-padding, +padding])

        w = xmin_xmax[1] - xmin_xmax[0]
        h = ymin_ymax[1] - ymin_ymax[0]
        start_x = np.abs(xmin_xmax[0])
        start_y = np.abs(ymin_ymax[0])
        # return the copy of sketch. you may use it.
        return [int(start_x), int(start_y), int(h), int(w)], sketch[:]

    def draw_sketch(sketch, window_name="sketch_visualize", padding=30,
                    thickness=2, random_color=True, draw_time=1, drawing=True):
        """
        Include drawing.
        Drawing under the guidance of positions and canvas's size given by canvas_size
        :param sketch: (n, 3) or (n, 4)
        :param window_name:
        :param padding:
        :param thickness:
        :param random_color:
        :param draw_time:
        :param drawing:
        :return: None
        """

        [start_x, start_y, h, w], sketch = canvas_size(sketch=sketch, padding=padding)
        canvas = np.ones((h, w, 3), dtype='uint8') * 255
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (0, 0, 0)
        pen_now = np.array([start_x, start_y])
        first_zero = False
        for stroke in sketch:
            delta_x_y = stroke[0:0 + 2]
            state = stroke[2]
            if first_zero:  # the first 0 in a complete stroke
                pen_now += delta_x_y
                first_zero = False
                continue
            cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
            if int(state) != 0:  # next stroke
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)
            pen_now += delta_x_y
            if drawing:
                cv2.imshow(window_name, canvas)
                key = cv2.waitKeyEx(draw_time)
                if key == 27:  # esc
                    cv2.destroyAllWindows()
                    exit(0)

        if drawing:
            key = cv2.waitKeyEx()
            if key == 27:  # esc
                cv2.destroyAllWindows()
                exit(0)
        # cv2.imwrite("./visualize.png", canvas)
        return canvas

    count = 0
    npz_all = du.get_allfiles(npz_root, 'npz')

    for index, fileName in enumerate(npz_all):
        print(f"|{index}|{fileName}|{fileName.split('_')[-1].split('.')[0]}|")
        # choose latin1 encoding because we make this dataset by python2.
        sketches = np.load(fileName, encoding="latin1", allow_pickle=True)
        # randomly choose one sketch in one .npz .
        for key in list(sketches.keys()):  # key 只有 arr_0
            # print(f"This key is {key}")
            # print(len(sketches[key]))
            count += len(sketches[key])
            number = random.randint(0, len(sketches[key]))
            sample_sketch: np.ndarray = sketches[key][number]
            # In this part.
            # remove the first line. because in this visualize code, we do not need absolute start-up position.
            # we get the start position by ourselves in func canvas_size().

            # ** In fold ./augm, please comment the under line.
            # because in augm datasets, the first line is not Absolute position.
            sample_sketch[0][0:3] = np.array([0, 0, 0], dtype=sample_sketch.dtype)

            # in cv2, data type INT is allowed.
            # if dataset is normalized, you can make sample_sketch larger.
            # if you run this code in a non-desktop server, drawing=False is necessary.
            sample_sketch = (sample_sketch * 1).astype("int")
            print(sample_sketch)
            draw_sketch(sample_sketch, drawing=True)
    print(count)


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


def vis_quickdraw(npz_file):
    sketch_all = fr.npz_read(npz_file)[0]
    for c_sketch in sketch_all:
        vis_sketch_data(c_sketch)


def vis_tensor_map(cuda_tensor, title=None, save_root=None, is_show=True):
    m, n = cuda_tensor.size()

    # 1. 将 CUDA Tensor 转换为 CPU 上的 NumPy 数组
    cpu_array = cuda_tensor.cpu().numpy()  # 关键步骤：数据从 GPU → CPU

    # 2. 绘制矩阵热力图
    plt.figure(figsize=(8, 4))  # 设置图像尺寸

    # 绘制热力图，cmap 指定颜色映射（如 'viridis'、'coolwarm' 等）
    plt.imshow(cpu_array, cmap='viridis', interpolation='nearest', aspect='auto')

    # 3. 自定义图像样式
    plt.title(title)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.xticks(range(n))
    plt.yticks(range(m))
    plt.colorbar()

    if save_root is not None:
        plt.savefig(save_root)

    if is_show:
        plt.show()

    plt.clf()
    plt.close()


if __name__ == '__main__':
    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Gear\8646fb6b0a7f42bb9d5036995471b6b0_1.txt', show_dot=True)

    # show_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32_no_mix_proc\110.txt', show_dot=True)

    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\toothbrush\18246.txt')
    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt_cls\train\alarm clock\85.txt', show_dot=True)
    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\16.txt')

    # ahead, ext = os.path.splitext(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\train\apple\177.txt')

    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk32_stkpnt32\train\Gear\0dd1520e215d8d4c3cdbfe889316ba33_4.txt')

    # --- vis
    # vis_sketch_folder()


    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e667cbb1491b6cf657d8627e60604c7c_3.txt'
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut\e667cbb1491b6cf657d8627e60604c7c_3.txt'
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Turbine\6cb038970ff89914aaf1010e1fae3505_3.txt'
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Sprocket\ee6719b6fd0110c3ffaba2c28a4a7e38_3.txt'
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Rivet\4c7f5c43de8782deda9d715bc6c0a804_4.txt'
    # the_file = r'D:\document\DeepLearning\DataSet\sketch_retrieval\test_dataset\sketches\airplane\n02691156_359-2.txt'

    # vis_sketch_orig(the_file, show_dot=True, dot_gap=3)


    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\1.txt', show_dot=False)

    # test()

    # vis_sketch_unified(r'')

    # vis_false_log(r'C:\Users\ChengXi\Downloads\false_instance.txt')

    # vis_sketch_orig(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Bearing\f973078416a6819866b86970c22ae8f9_4.txt', show_dot=True)

    # du.svg_to_txt(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\cup\5125.svg', r'C:\Users\ChengXi\Desktop\fig\asasas.txt')

    # a_svg_file = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\svg\apple\333.svg'
    # svg_fig = du.svg_read(a_svg_file)
    # vis_sketch_data(svg_fig)
    #
    # # vis_sketch_orig(r'C:\Users\ChengXi\Desktop\fig\asasas.txt')
    #
    # test_npz = r'D:\document\DeepLearning\DataSet\quickdraw\raw\apple.full.npz'
    # sketch_all = du.npz_read(test_npz)[0]
    #
    # for c_sketch in sketch_all:
    #     vis_sketch_data(c_sketch)

    # vis_quickdraw(fr'D:\document\DeepLearning\DataSet\quickdraw\raw\laptop.full.npz')

    # 有效的草图
    # the_file = r'D:\document\DeepLearning\DataSet\quickdraw\raw\airplane.full.npz'
    # the_file = r'D:\document\DeepLearning\DataSet\quickdraw\raw\angel.full.npz'
    # the_file = r'D:\document\DeepLearning\DataSet\quickdraw\raw\shark.full.npz'
    # the_file = r'D:\document\DeepLearning\DataSet\quickdraw\raw\bicycle.full.npz'
    # vis_quickdraw(the_file)

    vis_sketch_unified(r'D:\document\DeepLearning\DataSet\quickdraw\diffusion\apple_7_16\0.txt', 7, 16)

    pass


