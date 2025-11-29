from svgpathtools import svg2paths2
import numpy as np
import os
from tqdm import tqdm
import shutil
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import cv2
import json
from multiprocessing import Pool
from functools import partial
import random
import einops
import matplotlib.pyplot as plt

from data_utils import sketch_utils as du
from data_utils import sketch_file_read as fr
from data_utils import preprocess as pp
from encoders import spline as sp
import global_defs


def svg_to_s3(svg_path, txt_path, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up, delimiter=','):
    svg_data = fr.svg_read(svg_path, pen_down, pen_up)
    np.savetxt(txt_path, svg_data, delimiter=delimiter)


def svg_to_s3_ass(c_file, source_dir, target_dir):
    try:
        txt_path = c_file.replace(source_dir, target_dir).replace('svg', 'txt')
        svg_to_s3(c_file, txt_path)

    except:
        try:
            du.fix_svg_file(c_file)
            txt_path = c_file.replace(source_dir, target_dir).replace('svg', 'txt')
            svg_to_s3(c_file, txt_path)

        except:
            try:

                du.delete_lines_with_ampersand(c_file)
                txt_path = c_file.replace(source_dir, target_dir).replace('svg', 'txt')
                svg_to_s3(c_file, txt_path)

            except:
                try:
                    du.remove_svg_comments(c_file)
                    txt_path = c_file.replace(source_dir, target_dir).replace('svg', 'txt')
                    svg_to_s3(c_file, txt_path)

                except:
                    raise ValueError('err at ' + c_file)


def svg_to_s3_batched(source_dir, target_dir, workers=4):
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    du.create_tree_like(source_dir, target_dir)
    files_all = du.get_allfiles(source_dir, 'svg')

    worker_func = partial(svg_to_s3_ass,
                          source_dir=source_dir,
                          target_dir=target_dir
                          )

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, files_all),
            total=len(files_all),
            desc='svg to s3')
        )

    # files_all = du.get_allfiles(source_dir, 'svg')
    # for c_file in tqdm(files_all, total=len(files_all)):
    #     try:
    #         svg_to_txt(c_file, c_file.replace(source_dir, target_dir).replace('svg', 'txt'))
    #     except:
    #         print(f'trans failure: {c_file}')


def s3_to_svg(txt_file, svg_file, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, delimiter=',', stroke_width=2, stroke_color='black', canvas_size=800, padding=20, is_print_log=True):
    """
    将txt文件转化为svg文件，txt文件需要保存为每行(x, y, s)格式
    该函数用于配合sketch-a-net变形代码进行数据增强
    :param txt_file:
    :param svg_file:
    :param pen_up:
    :param pen_down:
    :return:
    """
    sketch = du.sketch_split(txt_file, pen_up, pen_down, delimiter)

    # 提取所有点
    all_points = np.vstack([s for s in sketch if len(s) > 0])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)

    min_x, min_y = min_xy
    max_x, max_y = max_xy
    width = max_x - min_x
    height = max_y - min_y

    # 可视区域大小
    usable_size = canvas_size - 2 * padding
    scale_factor = min(usable_size / width, usable_size / height) if width > 0 and height > 0 else 1.0

    # 第一个 translate 把图形移到 padding 位置
    trans_x = padding
    trans_y = padding + scale_factor * (max_y - min_y)  # y 方向是向下的

    # 构造 header
    header = f'''<?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg viewBox="0 0 {canvas_size} {canvas_size}" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <g fill="none" stroke="{stroke_color}" stroke-linecap="round" stroke-linejoin="round" stroke-width="{stroke_width}">
    <g transform="translate({trans_x:.4f},{trans_y:.4f}) scale({scale_factor:.4f}) translate({-min_x:.4f},{-max_y:.4f})">
    '''

    footer = '''</g>
    </g>
    </svg>
    '''

    def stroke_to_bezier_path(_stroke):
        """
        将一个二维点序列转换为 SVG 路径字符串，使用 C 命令（三次 Bézier 曲线）。
        简单策略：每 3 个点组成一段 Bézier，如果不足则做简化处理。
        """
        if len(_stroke) < 2:
            return ""  # 没有足够的点画路径

        d = f"M {_stroke[0][0]:.2f} {_stroke[0][1]:.2f}"

        i = 1
        while i + 2 < len(_stroke):
            p1, p2, p3 = _stroke[i], _stroke[i + 1], _stroke[i + 2]
            d += f" C {p1[0]:.2f} {p1[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f}, {p3[0]:.2f} {p3[1]:.2f}"
            i += 3

        # 若剩下 1~2 个点，退化为直线或重复控制点处理
        # if i < len(_stroke):
        #     remaining = _stroke[i:]
        #     if len(remaining) == 2:
        #         # 使用两个点，复制起点作为控制点
        #         d += f" C {remaining[0][0]:.2f} {remaining[0][1]:.2f}, {remaining[0][0]:.2f} {remaining[0][1]:.2f}, {remaining[1][0]:.2f} {remaining[1][1]:.2f}"
        #     elif len(remaining) == 1:
        #         d += f" L {remaining[0][0]:.2f} {remaining[0][1]:.2f}"

        return d

    # 构造 path 行
    path_lines = []
    for pathid, stroke in enumerate(sketch):
        if len(stroke) == 0:
            continue
        # d = f'M {stroke[0][0]:.2f} {stroke[0][1]:.2f}'
        # d += ''.join(f' L {x:.2f} {y:.2f}' for x, y in stroke[1:])

        d = stroke_to_bezier_path(stroke)
        path_lines.append(f'<path pathid="{pathid}" d="{d}"/>\n')

    # 写入
    with open(svg_file, 'w') as f:
        f.write(header)
        f.writelines(path_lines)
        f.write(footer)

    if is_print_log:
        print(f"✅ SVG 文件已保存：{svg_file}")
        print(
            f"📐 transform = translate({trans_x:.2f},{trans_y:.2f}) scale({scale_factor:.4f}) translate({-min_x:.2f},{-max_y:.2f})")


def s3_to_svg_batched(source_dir, target_dir):
    if os.path.exists(target_dir):
        user = input(f"目标文件夹已存在文件，删除它？(y / n)").strip()
        if user.lower() == 'y':
            print('delete dir' + target_dir)
            shutil.rmtree(target_dir)
        else:
            exit(0)
    os.makedirs(target_dir, exist_ok=True)

    # 在target_dir中创建与source_dir相同的目录层级
    du.create_tree_like(source_dir, target_dir)
    files_all = du.get_allfiles(source_dir, 'txt')

    for c_txt in tqdm(files_all):
        c_svg = c_txt.replace(source_dir, target_dir)
        c_svg = os.path.splitext(c_svg)[0] + '.svg'
        s3_to_svg(c_txt, c_svg, is_print_log=False)


def sketch_file_to_s5(root, max_length, coor_mode='ABS', is_shuffle_stroke=False):
    """
    将草图转换为 S5 格式，(x, y, s1, s2, s3)
    默认存储绝对坐标
    :param root:
    :param max_length:
    :param coor_mode: ['ABS', 'REL'], 'ABS': absolute coordinate. 'REL': relative coordinate [(x,y), (△x, △y), (△x, △y), ...].
    :param is_shuffle_stroke: 是否打乱笔划
    :return:
    """
    if isinstance(root, str):
        file_suffix = Path(root).suffix
        if file_suffix == '.txt':
            data_raw = fr.load_sketch_file(root)
        elif file_suffix == '.svg':
            data_raw = fr.svg_read(root)
        else:
            raise TypeError('error suffix')
    else:
        raise TypeError('error root type')

    # 打乱笔划
    if is_shuffle_stroke:
        stroke_list = np.split(data_raw, np.where(data_raw[:, 2] == global_defs.pen_up)[0] + 1)[:-1]
        random.shuffle(stroke_list)
        data_raw = np.vstack(stroke_list)

    # 多于指定点数则进行截断
    n_point_raw = len(data_raw)
    if n_point_raw > max_length:
        data_raw = data_raw[:max_length, :]

        # choice = np.random.choice(n_point_raw, max_length, replace=True)
        # data_raw = data_raw[choice, :]

    # [n_points, 3]
    data_raw = du.sketch_std(data_raw)

    # plt.plot(data_raw[:, 0], data_raw[:, 1])
    # plt.show()

    # 相对坐标
    if coor_mode == 'REL':
        coordinate = data_raw[:, :2]
        coordinate[1:] = coordinate[1:] - coordinate[:-1]
        data_raw[:, :2] = coordinate

    elif coor_mode == 'ABS':
        # 无需处理
        pass

    else:
        raise TypeError('error coor mode')

    c_sketch_len = len(data_raw)
    data_raw = torch.from_numpy(data_raw)

    data_cube = torch.zeros(max_length, 5, dtype=torch.float)
    mask = torch.zeros(max_length, dtype=torch.float)

    data_cube[:c_sketch_len, :2] = data_raw[:, :2]
    data_cube[:c_sketch_len, 2] = data_raw[:, 2]
    data_cube[:c_sketch_len, 3] = 1 - data_raw[:, 2]
    data_cube[-1, 4] = 1

    mask[:c_sketch_len] = 1

    return data_cube, mask


def quickdraw_to_mgt(root_npz, root_target, delimiter=',', select=(1000, 100, 100), is_random_select=False):
    """
    该函数主要用于从 npz 文件中获取 MGT 数据集

    该函数根据 QuickDraw 的 npz 文件编写，主要特征如下：
    1. 存储相对坐标
    2. 加载的字典包含三个键，分别是 'train', 'test', 'valid'
    如果你的 npz 文件不符合以上要求，请修改

    将创建如下文件夹：
    root_target
    ├─ train
    │   └─ npz_name
    │       ├─ 1.txt
    │       ├─ 2.txt
    │       ├─ 3.txt
    │       ...
    │
    ├─ test
    │   └─ npz_name
    │       ├─ 1.txt
    │       ├─ 2.txt
    │       ├─ 3.txt
    │       ...
    │
    └─ valid
        └─ npz_name
            ├─ 1.txt
            ├─ 2.txt
            ├─ 3.txt
            ...

    :param root_npz:
    :param root_target:
    :param delimiter: 保存 txt 文件时的分隔符
    :param select: 从 [train, test, valid] 分支中抽取的草图数 (数量来自 MGT). = None 则不选取
    :param is_random_select: 是否随机选取
    :return:
    """

    def _get_n_pnt_near(_sketch_list, _select, _pnt_base=35):
        """
        从一个草图的 list 中选择指定数量的点数最靠近 _pnt_base 的草图
        :param _sketch_list:
        :param _select:
        :param _pnt_base:
        :return:
        """
        # 按行数与 pnt_base 的绝对差值排序
        sorted_arrays = sorted(_sketch_list, key=lambda arr: abs(arr.shape[0] - _pnt_base))

        # 返回前 select 个
        return sorted_arrays[:_select]

    # 先读取数据
    # print('load data')
    std_train = fr.npz_read(root_npz, 'train')[0]
    std_test = fr.npz_read(root_npz, 'test')[0]
    std_valid = fr.npz_read(root_npz, 'valid')[0]

    if select is not None:
        sample_func = random.sample if is_random_select else _get_n_pnt_near

        std_train = sample_func(std_train, select[0])
        std_test = sample_func(std_test, select[1])
        std_valid = sample_func(std_valid, select[2])

    # 创建文件夹
    # print('create dirs')
    file_name = os.path.splitext(os.path.basename(root_npz))[0].replace('.', '_').replace(' ', '_')

    train_dir = os.path.join(root_target, 'train', file_name)
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(root_target, 'test', file_name)
    os.makedirs(test_dir, exist_ok=True)

    valid_dir = os.path.join(root_target, 'valid', file_name)
    os.makedirs(valid_dir, exist_ok=True)

    # 保存数据
    # print('save data')
    for idx, c_train in enumerate(std_train):
        c_train_filename = os.path.join(train_dir, f'{idx}.txt')
        np.savetxt(c_train_filename, c_train, delimiter=delimiter)

    for idx, c_test in enumerate(std_test):
        c_test_filename = os.path.join(test_dir, f'{idx}.txt')
        np.savetxt(c_test_filename, c_test, delimiter=delimiter)

    for idx, c_valid in enumerate(std_valid):
        c_valid_filename = os.path.join(valid_dir, f'{idx}.txt')
        np.savetxt(c_valid_filename, c_valid, delimiter=delimiter)


def quickdraw_to_mgt_batched(root_npz, root_target, is_random_select=True, workers=4):
    npz_all = du.get_allfiles(root_npz, 'npz')

    worker_func = partial(quickdraw_to_mgt, root_target=root_target, is_random_select=is_random_select)

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, npz_all),
            total=len(npz_all),
            desc='QuickDraw to MGT')
        )

    # for c_npz in tqdm(npz_all, total=len(npz_all)):
    #     quickdraw_to_mgt(c_npz, root_target, is_random_select=is_random_select)


def quickdraw_to_png(npz_file, save_root, n_save, linewidth=5, npz_tag='test', pen_up=global_defs.pen_up, pen_down=global_defs.pen_down):
    """
    用于将quickdraw的npz文件转化为png图片
    :param npz_file:
    :param n_save:
    :param save_root: 保存目录
    :return:
    """
    os.makedirs(save_root, exist_ok=True)

    skh_all = fr.npz_read(npz_file, npz_tag)[0]

    # 随机选取一些样本
    # skh_sel = np.random.choice(skh_all, size=n_save, replace=False)
    skh_sel = random.sample(skh_all, k=n_save)
    # 保存

    for idx, c_std_skh in tqdm(enumerate(skh_sel), total=n_save):
        # 最后一行最后一个数改为17，防止出现空数组
        c_std_skh[-1, 2] = pen_down

        # split all strokes
        strokes = np.split(c_std_skh, np.where(c_std_skh[:, 2] == pen_up)[0] + 1)

        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1], linewidth=linewidth)

        plt.axis('equal')
        plt.axis('off')
        plt.savefig(os.path.join(save_root, f'{idx}.png'))
        plt.clf()
        plt.close()


def s3_to_tensor_img(sketch, image_size=(224, 224), line_thickness=1, pen_up=global_defs.pen_up, coor_mode='ABS', save_path=None):
    """
    将 S3 草图转化为 Tensor 图片
    sketch: np.ndarray

    x1, y1, s1
    x2, y2, s2
    ...
    xn, yn, sn

    x, y 为绝对坐标
    s = 1: 下一个点属于当前笔划
    s = 0: 下一个点不属于当前笔划
    注意 Quickdraw 中存储相对坐标，不能直接使用

    :param sketch: 文件路径或者加载好的 [n, 3] 草图
    :param image_size:
    :param line_thickness:
    :param pen_up:
    :return: list(image_size), 224, 224 为预训练的 vit 的图片大小
    """
    assert coor_mode in ['REL', 'ABS']
    width, height = image_size

    if isinstance(sketch, str):
        points_with_state = fr.load_sketch_file(sketch)

    elif isinstance(sketch, np.ndarray):
        points_with_state = sketch

    else:
        raise TypeError('error sketch type')

    if coor_mode == 'REL':
        points_with_state[:, :2] = np.cumsum(points_with_state[:, :2], axis=0)

    # 1. 坐标归一化
    pts = np.array(points_with_state[:, :2], dtype=np.float32)
    states = np.array(points_with_state[:, 2], dtype=np.int32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diff_xy = max_xy - min_xy

    if np.allclose(diff_xy, 0):
        scale_x = scale_y = 1.0
    else:
        scale_x = (width - 1) / diff_xy[0] if diff_xy[0] > 0 else 1.0
        scale_y = (height - 1) / diff_xy[1] if diff_xy[1] > 0 else 1.0
    scale = min(scale_x, scale_y)

    pts_scaled = (pts - min_xy) * scale
    pts_int = np.round(pts_scaled).astype(np.int32)

    offset_x = (width - (diff_xy[0] * scale)) / 2 if diff_xy[0] > 0 else 0
    offset_y = (height - (diff_xy[1] * scale)) / 2 if diff_xy[1] > 0 else 0
    pts_int[:, 0] += int(round(offset_x))
    pts_int[:, 1] += int(round(offset_y))

    # 2. 创建白色画布
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 3. 笔划切分
    split_indices = np.where(states == pen_up)[0] + 1  # 下一个点是新笔划，所以+1
    strokes = np.split(pts_int, split_indices)

    # 4. 绘制每条笔划
    for stroke in strokes:
        if len(stroke) >= 2:  # 至少2个点才能画线
            stroke = stroke.reshape(-1, 1, 2)
            cv2.polylines(img, [stroke], isClosed=False, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    # 5. 转为归一化float32 Tensor
    tensor_img = torch.from_numpy(img).float() / 255.0

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return tensor_img


def s3_to_stk_ass(std_file, source_dir, target_dir, preprocess_func, delimiter, pen_up, pen_down):
    try:
        c_target_file = std_file.replace(source_dir, target_dir)

        target_skh_STK = preprocess_func(std_file, pen_up=pen_up, pen_down=pen_down, delimiter=delimiter)
        target_skh_STK = einops.rearrange(target_skh_STK, 's sp c -> (s sp) c')

        if len(target_skh_STK) == global_defs.n_skh_pnt:
            np.savetxt(c_target_file, target_skh_STK, delimiter=delimiter)
        else:
            print(f'error occurred, skip file: {std_file}')
    except:
        print(f'error occurred, skip file: {std_file}')


def s3_to_stk_batched(source_dir, target_dir, preprocess_func=pp.preprocess_stk, delimiter=',', workers=4, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down):
    """
    将 source_dir 内的 std 草图转化为 STK 草图
    std 草图：每行为 [x, y, s]，行数不固定
    将在 target_dir 内创建与 source_dir 相同的层级结构
    文件后缀为 .STK
    :param source_dir:
    :param target_dir:
    :param preprocess_func:
    :param delimiter:
    :param workers: 处理进程数
    :return:
    """
    # 在 target_dir 内创建与 source_dir 相同的文件夹层级结构
    print('create dirs')
    du.create_tree_like(source_dir, target_dir)

    # 获得source_dir中的全部文件
    files_all = du.get_allfiles(source_dir, 'txt')

    worker_func = partial(s3_to_stk_ass,
                          source_dir=source_dir,
                          target_dir=target_dir,
                          preprocess_func=preprocess_func,
                          delimiter=delimiter,
                          pen_up=pen_up,
                          pen_down=pen_down
                          )

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, files_all),
            total=len(files_all),
            desc='QuickDraw to MGT')
        )


def npz_to_stk_ass(idx_skh, stk_root_inner, preprocess_func, delimiter, is_order_stk):
    idx, c_skh = idx_skh

    try:
        c_target_file = os.path.join(stk_root_inner, f'{idx}.txt')

        target_skh_STK = preprocess_func(c_skh, is_order_stk=is_order_stk)
        target_skh_STK = einops.rearrange(target_skh_STK, 's sp c -> (s sp) c')

        if len(target_skh_STK) == global_defs.n_skh_pnt:

            if target_skh_STK.shape[1] == 3:  # STK2
                # fmt = ["%.6f", "%.6f", "%d"]
                np.savetxt(c_target_file, target_skh_STK, delimiter=delimiter, fmt="%.6f")
            else:
                np.savetxt(c_target_file, target_skh_STK, delimiter=delimiter, fmt="%.6f")
        else:
            print(f'error occurred, skip instance: {idx}')

    except Exception as e:
        print(f'error occurred, skip instance: {idx}, exception: {e}')


def npz_to_stk_file(npz_file, stk_root, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, preprocess_func=pp.preprocess_stk, delimiter=',', workers=4, is_order_stk=True, add_savefolder_str=''):
    """
    将npz文件转化为stk草图并保存
    :param npz_file:
    :param stk_root:
    :param n_stk:
    :param n_stk_pnt:
    :param preprocess_func:
    :param delimiter:
    :param workers:
    :param is_order_stk:
    :param add_savefolder_str: 附加的自定义保存文件夹名附加字符串
    :return:
    """
    class_name = os.path.basename(npz_file).split('.')[0]

    if is_order_stk:
        stk_root_inner = os.path.join(stk_root, f'{class_name}_order_stk_{n_stk}_{n_stk_pnt}' + add_savefolder_str)
    else:
        stk_root_inner = os.path.join(stk_root, f'{class_name}_stk_{n_stk}_{n_stk_pnt}' + add_savefolder_str)

    os.makedirs(stk_root_inner, exist_ok=True)

    skh_all = fr.npz_read(npz_file, 'train')[0]

    worker_func = partial(npz_to_stk_ass,
                          stk_root_inner=stk_root_inner,
                          preprocess_func=preprocess_func,
                          delimiter=delimiter,
                          is_order_stk=is_order_stk
                          )

    param_input = list(enumerate(skh_all))

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, param_input),
            total=len(param_input),
            desc='QuickDraw to MGT')
        )


def stroke_list_to_s3(stroke_list, pen_down=global_defs.pen_down, pen_up=global_defs.pen_up):
    """
    将笔划数组转化为 s3 格式的 ndarray
    :param stroke_list:
    :param pen_down:
    :param pen_up:
    :return:
    """
    # 在每个点的末尾加上笔划状态
    stroke_list_np = []
    for c_stk in stroke_list:

        # 先构建一个全为 pen_down 的ndarray
        n = len(c_stk)
        ones_col = np.full((n, 1), pen_down, dtype=c_stk.dtype)

        # 将最后一个设置为 pen_up
        ones_col[-1, 0] = pen_up

        # 拼接到 (x, y) 上
        c_stk = np.hstack((c_stk, ones_col))
        stroke_list_np.append(c_stk)

    stroke_list_np = np.vstack(stroke_list_np)
    return stroke_list_np


def s3_to_fix_point_s3(s3_data, n_point=global_defs.n_skh_pnt):
    """
    将 S3 草图中的点数采样到固定数值
    根据长度分配点数
    保留相对采样密度

    :param s3_data:
    :param n_point:
    :return:
    """
    if isinstance(s3_data, str):
        s3_data = fr.load_sketch_file(s3_data)
        stroke_list = du.sketch_split(s3_data)

    elif isinstance(s3_data, np.ndarray):
        stroke_list = du.sketch_split(s3_data)

    elif isinstance(s3_data, list):
        stroke_list = s3_data

    else:
        raise TypeError('error s3 data format')

    # 根据长度分配点数，每个笔划至少分配两个点
    stk_lens = []
    sketch_length = 0.0

    for c_stk in stroke_list:
        c_length = du.stroke_length(c_stk)

        stk_lens.append(c_length)
        sketch_length += c_length

    point_assign = []
    for c_len in stk_lens:
        c_n_point = round((c_len / sketch_length) * n_point)

        if c_n_point < 2:
            c_n_point = 2

        point_assign.append(c_n_point)

    # 保留密度采样
    sampled_stk = []
    for c_stk, c_n_point in zip(stroke_list, point_assign):
        c_sampled = sp.sample_keep_dense(c_stk, c_n_point)
        sampled_stk.append(c_sampled)

    sampled_stk = stroke_list_to_s3(sampled_stk)
    return sampled_stk


def s3_to_fix_point_s3_ass(s3_file, source_dir, target_dir, n_pnt, delimiter=','):
    try:
        target_pix_pnt_s3 = s3_to_fix_point_s3(s3_file, n_pnt)

        c_target_file = s3_file.replace(source_dir, target_dir)
        np.savetxt(c_target_file, target_pix_pnt_s3, delimiter=delimiter)

    except:
        print(f'error occurred, skip file: {s3_file}')


def s3_to_fix_point_s3_batched(source_dir, target_dir, n_pnt=global_defs.n_skh_pnt, workers=4):
    # 在 target_dir 内创建与 source_dir 相同的文件夹层级结构
    print('create dirs')
    du.create_tree_like(source_dir, target_dir)

    # 获得source_dir中的全部文件
    files_all = du.get_allfiles(source_dir, 'txt')

    worker_func = partial(s3_to_fix_point_s3_ass,
                          source_dir=source_dir,
                          target_dir=target_dir,
                          n_pnt=n_pnt
                          )

    with Pool(processes=workers) as pool:
        data_trans = list(tqdm(
            pool.imap(worker_func, files_all),
            total=len(files_all),
            desc='s3 to fix point number s3')
        )


if __name__ == '__main__':
    # npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\apple.full.npz',
    #                 r'D:\document\DeepLearning\DataSet\quickdraw\diffusion')
    #
    # npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\moon.full.npz',
    #                 r'D:\document\DeepLearning\DataSet\quickdraw\diffusion')
    #
    npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\book.full.npz',
                    r'D:\document\DeepLearning\DataSet\quickdraw\diffusion', preprocess_func=pp.preprocess_stk2, delimiter=' ')
    #
    # npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\shark.full.npz',
    #                 r'D:\document\DeepLearning\DataSet\quickdraw\diffusion')
    #
    # npz_to_stk_file(r'D:\document\DeepLearning\DataSet\quickdraw\raw\angel.full.npz',
    #                 r'D:\document\DeepLearning\DataSet\quickdraw\diffusion')

    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\apple.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\apple', 1000)
    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\moon.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\moon', 1000)
    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\book.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\book', 1000)
    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\shark.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\shark', 1000)
    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\angel.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\angel', 1000)
    # quickdraw_to_png(r'D:\document\DeepLearning\DataSet\quickdraw\raw\bicycle.full.npz', r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\bicycle', 1000)



    # s3_to_stk_batched(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketches_s3',
    #                    rf'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketches_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')


    # txt_to_svg(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Bearing\00b11be6f26c85ca85f84daf52626b36_1.txt', r'E:\document\DeepLearning\sketch-specific-data-augmentation\convert.svg')

    # svg_to_txt(r'C:\Users\ChengXi\Desktop\0a6d329de93891ee4b8ecfd8b08feee7_2_1.svg', r'C:\Users\ChengXi\Desktop\0a6d329de93891ee4b8ecfd8b08feee7_2_1.txt')

    # s3_sample_to_specific_point(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy_other_files\sketches_svg\airplane\n02691156_394-2.svg')

    # atensor = image_to_vector_strokes(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketches_png\airplane\n02691156_394-2.png')
    # plt.imshow(atensor[1], cmap='gray')

    # img = cv2.imread(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketches_png\airplane\n02691156_394-2.png', cv2.IMREAD_GRAYSCALE)
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(r"C:\Users\ChengXi\Desktop\60mm20250708\sketch.bmp", binary)

    # cmpx_svg = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy_other_files\sketches_svg\airplane\n02691156_394-2.svg'
    # modify_stroke_width_for_black_strokes(cmpx_svg, r'C:\Users\ChengXi\Desktop\60mm20250708\sketch.svg', 2)

    # image_to_vector_strokes(r'C:\Users\ChengXi\Desktop\60mm20250708\sketch.png', visualize=True)

    # img_to_bmp(r'C:\Users\ChengXi\Desktop\60mm20250708\sketch.png', r'C:\Users\ChengXi\Desktop\60mm20250708\sketch.bmp')

    # raster_to_vector_sketch(r'C:\Users\ChengXi\Desktop\60mm20250708\sketch.png', visualize=True)



    # strokes = du.sketch_split(atensor)
    #
    # for c_stk in strokes:
    #     plt.plot(c_stk[:, 0], -c_stk[:, 1])
    #
    # plt.axis('equal')
    # plt.show()

    # svg_to_txt(cmpx_svg, r'C:\Users\ChengXi\Desktop\60mm20250708\transed.txt')

    # s3_to_fix_point_s3(r'C:\Users\ChengXi\Desktop\60mm20250708\transed.txt')

    # svg_to_s3_batched(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketches_svg_raw', r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketch_txt')

    # s3_to_fix_point_s3_batched(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketches_s3', rf'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3_{global_defs.n_skh_pnt}')

    # atensor = s3_to_tensor_img(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3_352\airplane\n02691156_196-5.txt', line_thickness=2, save_path=r'C:\Users\ChengXi\Desktop\60mm20250708\rel_skh.png')

    # s3_to_svg(r'C:\Users\ChengXi\Desktop\cstnet2\testsvg.txt', r'C:\Users\ChengXi\Desktop\cstnet2\testsvg.svg', 0, 1)

    # s3_to_svg_batched(r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all', r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_svg_all')
    pass

