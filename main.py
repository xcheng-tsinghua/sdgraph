import torch
import torch.nn as nn
import svgpathtools
from svgpathtools import svg2paths2

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import splprep, splev

# import data_utils.sketch_vis as vis
# from data_utils.sketch_utils import svg_to_txt


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


def curve_fit():
    # 生成圆形数据
    theta = np.linspace(0, 2 * np.pi, 50)
    x = np.cos(theta)
    y = np.sin(theta)

    # 添加一些噪声
    x += np.random.normal(0, 0.1, size=x.shape)
    y += np.random.normal(0, 0.1, size=y.shape)

    # 使用 splprep 进行参数化样条拟合
    tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度

    # 生成拟合曲线
    new_u = np.linspace(0, 1, 100)
    new_x, new_y = splev(new_u, tck)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o', label='原始点')
    plt.plot(new_x, new_y, '-', label='拟合曲线')
    plt.legend()
    plt.axis('equal')  # 保持比例
    plt.show()


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



if __name__ == '__main__':
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

    def hausdorff_distance(curves1, curves2):
        # 计算距离矩阵，形状为 (a, b)
        dist_matrix = torch.cdist(curves1, curves2, p=2)  # p=2表示欧氏距离

        # 对于curves1中的每条曲线，找到它到curves2的最小距离
        min_dist_curves1 = dist_matrix.min(dim=2)[0]

        # 对于curves2中的每条曲线，找到它到curves1的最小距离
        min_dist_curves2 = dist_matrix.min(dim=1)[0]

        # 计算Hausdorff距离
        return max(min_dist_curves1.max(), min_dist_curves2.max())


    # 示例数据
    curves1 = torch.rand(3, 8, 2)
    curves2 = torch.rand(4, 9, 2)

    # 计算Hausdorff距离
    distance = hausdorff_distance(curves1, curves2)
    print(f"Hausdorff Distance: {distance.item()}")
