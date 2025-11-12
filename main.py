import random
import math
import numpy
import torch
import torch.nn as nn
import svgpathtools
from svgpathtools import svg2paths2
import pandas as pd
import ruptures as rpt
import json
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import splprep, splev
from encoders_3rd.vit import VITFinetune
from encoders_3rd import sketch_rnn

# import data_utils.sketch_vis as vis
# from data_utils.sketch_utils import svg_to_txt
import data_utils.sketch_utils as skutils


def curve_fit(x, y, k):
    # 使用 splprep 进行参数化样条拟合
    tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度

    # 生成拟合曲线
    new_u = np.linspace(0, 1, k)
    new_x, new_y = splev(new_u, tck)

    return new_x, new_y


def down_sample_and_up_sample():
    # 输入图像：batch_size=1, channels=3 (RGB), height=4, width=8
    input_image = torch.randn(1, 3, 4, 64)

    # 定义 Conv2d
    conv = nn.Conv2d(
        in_channels=3,  # 输入通道数 (RGB)
        out_channels=3,  # 输出通道数 (保持通道数不变)
        kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
        stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
        padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
    )

    # 前向传播
    output_image = conv(input_image)

    conv_transpose = nn.ConvTranspose2d(
        in_channels=3,  # 输入通道数
        out_channels=3,  # 输出通道数
        kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
        stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
        padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
    )

    up_image = conv_transpose(output_image)

    conv_transpose_plus2 = nn.ConvTranspose2d(
        in_channels=3,  # 输入通道数
        out_channels=3,  # 输出通道数
        kernel_size=(1, 3),  # 卷积核大小：1x2，仅在宽度方向扩展
        # stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
        # padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
    )

    plus2 = conv_transpose_plus2(up_image)

    print("输入形状:", input_image.shape)  # torch.Size([1, 3, 4, 8])
    print("输出形状:", output_image.shape)  # torch.Size([1, 3, 4, 4])
    print("上采样形状:", up_image.shape)  # torch.Size([1, 3, 4, 4])
    print("+2形状:", plus2.shape)  # torch.Size([1, 3, 4, 4])


# def curve_fit():
#     # 生成圆形数据
#     theta = np.linspace(0, 2 * np.pi, 50)
#     x = np.cos(theta)
#     y = np.sin(theta)
#
#     # 添加一些噪声
#     x += np.random.normal(0, 0.1, size=x.shape)
#     y += np.random.normal(0, 0.1, size=y.shape)
#
#     # 使用 splprep 进行参数化样条拟合
#     tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度
#
#     # 生成拟合曲线
#     new_u = np.linspace(0, 1, 100)
#     new_x, new_y = splev(new_u, tck)
#
#     # 可视化
#     plt.figure(figsize=(6, 6))
#     plt.plot(x, y, 'o', label='原始点')
#     plt.plot(new_x, new_y, '-', label='拟合曲线')
#     plt.legend()
#     plt.axis('equal')  # 保持比例
#     plt.show()


def find_files_with_line_count_not_equal(directory, k):
    """
    找出指定目录中所有行数不为 k 的 txt 文件。

    :param directory: 要搜索的目录路径
    :param k: 要比较的行数
    :return: 行数不为 k 的文件列表
    """
    files_with_wrong_line_count = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # 只处理 .txt 文件
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    line_count = sum(1 for _ in file)  # 计算行数
                if line_count != k:
                    files_with_wrong_line_count.append(filepath)
            except Exception as e:
                print(f"无法处理文件 {filepath}: {e}")

    return files_with_wrong_line_count


def subset_search():
    def idx_step(idx_list: list, max_idx):
        """
        根据当前索引计算下一个索引
        :param idx_list:
        :param max_idx: 目标数组长度
        :return:
        """
        # 防止对参数进行修改
        idx_local = idx_list.copy()

        # 插入辅助索引，以使最后一个索引和其他索引有相同的判断条件
        idx_local.append(max_idx)

        # 定位到最初尝试移动的索引
        last_pointer = len(idx_local) - 2
        last_pointer_max = len(idx_local) - 2

        # 当前状态下索引课否移动
        is_success = False

        while True:
            # 如果最后一个不能向前，尝试移动前一位
            if (idx_local[last_pointer] + 1) == idx_local[last_pointer + 1]:
                last_pointer = last_pointer - 1

            else:
                idx_local[last_pointer] = idx_local[last_pointer] + 1

                # 不移动最后一个索引时，需要将当前移动的索引及之后的索引构建成自然升序
                if last_pointer != last_pointer_max:
                    former_part = idx_local[:last_pointer]

                    len_later = last_pointer_max - last_pointer + 1
                    began_later = idx_local[last_pointer]
                    later_part = list(range(began_later, began_later + len_later + 1))

                    idx_local = former_part + later_part

                is_success = True
                break

            # 全部不能移动时跳出循环
            if last_pointer < 0:
                break

        # 删除辅助索引
        idx_local.pop()

        return is_success, idx_local


    def search_idx(k=3):
        idx = list(range(k))

        all_idx = [[]]

        for i in range(1, len(idx) + 1):
            c_idx = list(range(i))
            all_idx.append(c_idx.copy())

            while True:
                c_res = idx_step(c_idx, k)

                if not c_res[0]:
                    break
                else:
                    c_idx = c_res[1]
                    all_idx.append(c_res[1].copy())

        return all_idx

    def search_sub_set(target: list):
        all_idx = search_idx(len(target))

        all_vals = []
        for c_idx in all_idx:
            c_val = [target[i] for i in c_idx]
            all_vals.append(c_val)

        return all_vals

    all_sub_set = search_sub_set([0, 1, 2, 3, 4, 5])

    for c_subset in all_sub_set:
        print(c_subset)
    print(len(all_sub_set))


def vis_stk_score():
    # 读取文件
    data = np.loadtxt(r'C:/Users/ChengXi/Desktop/sketch_test/scores.txt', delimiter=',')

    S1, S2, S3, X, Y = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    x_v = []
    y_v = []
    c1_v = []
    c2_v = []
    c3_v = []
    c_seg = []

    x_seg = []
    y_seg = []

    for i in range(len(S1)):
        x_seg.append(X[i])
        y_seg.append(Y[i])

        if S1[i] == 0 and S2[i] == 0 and S3[i] == 0:
            c_seg.append(1.0)
        elif S1[i] == 0.5 and S2[i] == 0.5 and S3[i] == 0.5:
            c_seg.append(0.5)
        else:
            c1_v.append(S1[i])
            c2_v.append(S2[i])
            c3_v.append(S3[i])
            x_v.append(X[i])
            y_v.append(Y[i])

            c_seg.append(0.0)

    scatter1 = axes[0].scatter(x_v, y_v, c=c1_v, cmap='viridis', edgecolor='none')
    axes[0].set_title("Scatter plot colored by S1")
    fig.colorbar(scatter1, ax=axes[0], label='S1')

    scatter2 = axes[1].scatter(x_v, y_v, c=c2_v, cmap='plasma', edgecolor='none')
    axes[1].set_title("Scatter plot colored by S2")
    fig.colorbar(scatter2, ax=axes[1], label='S2')

    scatter3 = axes[2].scatter(x_v, y_v, c=c3_v, cmap='coolwarm', edgecolor='none')
    axes[2].set_title("Scatter plot colored by S3")
    fig.colorbar(scatter3, ax=axes[2], label='S3')

    # 显示分割点
    scatter4 = axes[3].scatter(x_seg, y_seg, c=c_seg, cmap='viridis', edgecolor='none')
    axes[3].set_title("Splits")
    fig.colorbar(scatter4, ax=axes[3], label='splits')

    axes[4].plot(c1_v, c2_v)
    x_base = [c1_v[0], c1_v[-1]]
    y_base = [max(c2_v) * 0.8, max(c2_v) * 0.8]
    axes[4].plot(x_base, y_base)

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def detect_and_plot_square_wave(xlsx_file, sheet_name="13", pen=10):
    """
    从指定的 Excel 文件和表单中读取波形数据，检测方波并画图。

    参数说明：
    ----------
    xlsx_file : str
        Excel 文件路径（后缀 .xlsx）。
    sheet_name : str
        要读取的表名，默认为 "Sheet1"。
    pen : float
        ruptures 中的惩罚项，越大分段越少，默认为 10。

    使用示例：
    ----------
    detect_and_plot_square_wave("data.xlsx", sheet_name="Signal", pen=8)
    """

    # 1. 读取 Excel 数据（假设第 1 行是表头，实际数据从第 2 行开始）
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=0)

    # 如果你的 Excel 确实在第 2 行开始才是数据，可以通过 df.iloc 进一步裁剪
    df = df.iloc[1:, :]  # 如果确实要跳过第一行之外的更多行，可自行调整

    # 2. 提取 x、y 坐标 (这里假设第 1 列是 x, 第 2 列是 y)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # 3. 使用 ruptures 进行变点检测（以 Pelt 算法为例）
    model = rpt.Pelt(model="l2").fit(y)
    # 根据 pen 值寻找变点，返回的 result 列表包含所有分段末端的索引（包含 len(y)）
    result = model.predict(pen=pen)

    # 4. 构造方波：将每一段用该段的均值替代
    segData = np.zeros_like(y)
    start_idx = 0
    for end_idx in result:
        # 注意 end_idx 是分段末端+1 的位置，因此 segData[start_idx:end_idx]
        # 包含了该段的所有采样点
        segData[start_idx:end_idx] = y[start_idx:end_idx].mean()
        start_idx = end_idx

    # 5. 画图
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Raw Data", linewidth=1)
    plt.plot(x, segData, label="Detected Square Wave", linewidth=2)
    plt.title("Original Wave vs. Detected Square Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

    return x, y, segData


if __name__ == '__main__':

    sketch_rnn.sketch_rnn_proj()

    # curve_fit()
    # print(find_files_with_line_count_not_equal(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\banana_stk5_stkpnt32', 160))

    # alist = [1, 2, 3, 4]
    # val0 = alist[0]
    #
    # alist[0] = 3
    #
    # print(val0)

    # x = torch.tensor([[1.0, 2.0, 3.0],
    #                   [4.0, 5.0, 6.0]])
    #
    # x_normalized_dim0 = torch.nn.functional.normalize(x, dim=0)
    # print(x_normalized_dim0)
    # print(x_normalized_dim0.pow(2).sum(0))

    # svg_file = r'D:\document\DeepLearning\DataSet\TU-Berlin\sketches\airplane\1.svg'  # 输入的SVG文件路径
    # output_file = r'C:\Users\ChengXi\Desktop\1.txt'  # 输出的文本文件路径
    #
    # svg_to_txt(svg_file, output_file)
    # vis.vis_sketch_orig(output_file)

    # vis.vis_sketch_orig(r'D:\document\DeepLearning\DataSet\TU_Berlin_txt\armchair\530.txt')

    # def hausdorff_distance(curves1, curves2):
    #     # 计算距离矩阵，形状为 (a, b)
    #     dist_matrix = torch.cdist(curves1, curves2, p=2)  # p=2表示欧氏距离
    #
    #     # 对于curves1中的每条曲线，找到它到curves2的最小距离
    #     min_dist_curves1 = dist_matrix.min(dim=2)[0]
    #
    #     # 对于curves2中的每条曲线，找到它到curves1的最小距离
    #     min_dist_curves2 = dist_matrix.min(dim=1)[0]
    #
    #     # 计算Hausdorff距离
    #     return max(min_dist_curves1.max(), min_dist_curves2.max())
    #
    #
    # # 示例数据
    # curves1 = torch.rand(3, 8, 2)
    # curves2 = torch.rand(4, 9, 2)
    #
    # # 计算Hausdorff距离
    # distance = hausdorff_distance(curves1, curves2)
    # print(f"Hausdorff Distance: {distance.item()}")

    # mcba_cat = skutils.get_subdirs(r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCB_A\train')
    # mcbb_cat = skutils.get_subdirs(r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCB_B\train')
    #
    # set_mcba = set(mcba_cat)
    # set_mcbb = set(mcbb_cat)
    #
    # # 相同元素（交集）
    # common_elements = list(set_mcba & set_mcbb)
    #
    # # 不同元素（对称差集）
    # different_elements = list(set_mcba ^ set_mcbb)
    #
    # print(set_mcba)
    # print(set_mcbb)
    #
    #
    # print("相同元素:", common_elements)
    # print("不同元素:", different_elements)

    # vis_stk_score()

    # detect_and_plot_square_wave(r'C:\Users\ChengXi\Desktop\wave_data.xlsx')

    # 示例数组
    # all_preds = np.array([1, 2, 3, 4, 5])
    # all_labels = np.array([1, 2, 0, 4, 0])
    #
    # # 找出两个数组不同的元素的索引
    # diff_indices = np.where(all_preds != all_labels)[0]
    #
    # print("不同的元素索引位置:", diff_indices)

    # atensor = torch.Tensor([[True, True, False, True, True, False], [True, True, True, True, True, False]])
    # # print(atensor[::2])
    # print(~atensor.max(0)[0].bool())

    # print(torch.max(torch.tensor([1, 2, 3, 4, float('nan')])))

    # atensor = torch.ones(2, 4)
    #
    # print(atensor)
    # print(atensor.max(1)[1])

    # arr = np.array([
    #     [1.0, 2.0, 3.0],
    #     [4.0, 5.0, -1.0],
    #     [7.0, 8.0, 0.5],
    #     [9.0, 0.0, -2.0]
    # ])
    #
    # # 使用布尔索引过滤掉 z < 0 的点
    # filter_idx = arr[:, 2] >= 0
    # filtered_arr = arr[filter_idx]
    #
    # print(filtered_arr)

    # alist = []
    #
    # for i in range(10):
    #     alist.append(numpy.zeros([13, 2]))
    #
    # print(alist)
    # print(np.array(alist).shape)

    # anda = np.ones([10, 2])
    #
    # adad = np.cumsum(anda, axis=0)
    #
    # print(adad)
    # print(adad.shape)

    # test_list = list(range(10))
    # print(test_list)
    # random.shuffle(test_list)
    # print(test_list)
    #
    # print(math.ceil(0.21 * 10))

    # trst_tebser = torch.rand(3, 27, 8, 10)
    # conv_layer = nn.Conv2d(27, 27, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    # print(conv_layer(trst_tebser).size())



    # atensor = torch.rand(5, 3, 224, 224)
    #
    # amodel = VITFinetune(10)
    # print(amodel(atensor).size())

    # with open('bks/data.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # keys = list(data.keys())
    # values = list(data.values())
    #
    # print('keys = [', ' '.join(str(k) for k in keys), '];')
    # print('values = [', ' '.join(str(v) for v in values), '];')
    #
    #
    # # 将字符串类型的键转换为整数类型
    # data_int_keys = {int(k): v for k, v in data.items()}
    #
    # # 提取键和值
    # x = list(data.keys())
    # y = list(data.values())
    #
    # # 创建图表
    # plt.figure(figsize=(8, 5))
    # plt.bar(x, y)
    # plt.xlabel('n points in sketch')
    # plt.ylabel('number of sketches')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # 读取灰度图
    # img = cv2.imread('C:/Users/ChengXi/Desktop/figure.png', cv2.IMREAD_GRAYSCALE)
    # # 高斯平滑降噪
    # blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    # # Canny 检测：低阈值50，高阈值150
    # edges = cv2.Canny(blur, 50, 150)
    # edges = cv2.bitwise_not(edges)
    # cv2.imwrite('C:/Users/ChengXi/Desktop/edge.png', edges)

    # astr = 'n02694662_17391-2.svg'
    # print(astr.split('.'))



    pass




