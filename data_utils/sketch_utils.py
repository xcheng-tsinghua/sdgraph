import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from svgpathtools import svg2paths2
import warnings
import shutil
import math
from tqdm import tqdm
import requests
import torch

import global_defs
import encoders.spline as sp


def tmp_vis_sketch_list(strokes, show_dot=False):
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('on')
    plt.axis('equal')
    plt.show()


def n_sketch_pnt(sketch) -> int:
    """
    返回草图中的点数
    :param sketch:
    :return:
    """
    if isinstance(sketch, np.ndarray):
        return len(sketch)

    elif isinstance(sketch, list):
        pnt_count = 0
        for c_stk in sketch:
            pnt_count += len(c_stk)

        return pnt_count

    elif isinstance(sketch, str):
        sketch_data = np.loadtxt(sketch, delimiter=',')
        return len(sketch_data)

    else:
        raise TypeError('Unknown sketch type')


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    # 确定矩阵的大小（假设最大值为5，因此矩阵大小为6x6）
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    # 遍历 list1 和 list2 并更新矩阵
    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    # 使用 Matplotlib 可视化矩阵
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    try:
        plt.savefig(save_name)
    except:
        warnings.warn('can not save confusion matrix, for save path is not exist: ', save_name)
    plt.close()


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:

            if suffix is not None:
                if file.split('.')[-1] == suffix:
                    if filename_only:
                        current_filepath = file
                    else:
                        current_filepath = str(os.path.join(root, file))
                    filepath_all.append(current_filepath)

            else:
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


def translate_class_folder(root=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt\train'):
    """
    root
    ├─ bearing (folder)
    ├─ bolt (folder)
    ├─ ...
    └─ washer (folder)

    """
    terms_mapping = {
        '垫圈': 'Washer',
        '堵头': 'Plug',
        '带轮': 'Pulley',
        '弹簧': 'Spring',
        '挡圈': 'Retaining_ring',
        '接头': 'Joint',
        '法兰': 'Flange',
        '涡轮': 'Turbine',
        '脚轮': 'Caster',
        '螺柱': 'Stud',
        '螺栓': 'Bolt',
        '螺母': 'Nut',
        '螺钉': 'Screw',
        '衬套': 'Bushing',
        '轴承': 'Bearing',
        '铆钉': 'Rivet',
        '链轮': 'Sprocket',
        '销': 'Pin',
        '键': 'Key',
        '间隔器': 'Spacer',
        '阀门': 'Valve',
        '风扇': 'Fan',
        '齿轮': 'Gear'
    }

    sub_dirs = get_subdirs(root)

    for c_name in sub_dirs:
        new_name = terms_mapping[c_name]
        os.rename(os.path.join(root, c_name), os.path.join(root, new_name))


def sketch_std(sketch):
    """
    将草图质心移动到原点，范围归一化为 [-1, 1]^2
    :param sketch: [n_point, s]
    :return:
    """
    assert isinstance(sketch, np.ndarray)

    coordinates = sketch[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    mean_coor = np.mean(coordinates, axis=0)
    mean_coor = np.expand_dims(mean_coor, 0)
    coordinates = coordinates - mean_coor  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch[:, :2] = coordinates
    return sketch


def svg_to_txt(svg_path, txt_path):
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    strokes = []

    for path, attr in zip(paths, attributes):
        if len(path) == 0:
            continue

        # 分割子路径（处理M/m移动命令）
        subpaths = []
        current_subpath = []

        for segment in path:
            if segment.start != (current_subpath[-1].end if current_subpath else None):
                if current_subpath:
                    subpaths.append(current_subpath)
                current_subpath = []
            current_subpath.append(segment)

        if current_subpath:
            subpaths.append(current_subpath)

        # 处理每个子路径
        for subpath in subpaths:
            points = []
            # 添加第一个线段的起点
            points.append((subpath[0].start.real, subpath[0].start.imag))

            # 添加所有线段的终点
            for segment in subpath:
                points.append((segment.end.real, segment.end.imag))

            strokes.append(points)

    with open(txt_path, 'w') as f:
        for stroke_idx, stroke in enumerate(strokes):
            for i, (x, y) in enumerate(stroke):
                # 笔划状态判断（当前笔划的最后一个点标记s=0）
                s = 0 if (i == len(stroke) - 1) and (stroke_idx != len(strokes) - 1) else 1

                # 写入文件，保留3位小数
                f.write(f"{round(x, 3):.3f},{round(y, 3):.3f},{s}\n")


def svg_to_txt_batched(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)

    files_all = get_allfiles(source_dir, 'svg')

    for c_file in tqdm(files_all, total=len(files_all)):
        try:
            svg_to_txt(c_file, c_file.replace(source_dir, target_dir).replace('svg', 'txt'))
        except:
            print(f'trans failure: {c_file}')


def std_unify(std_root: str, min_pnt: int = global_defs.n_stk * 2, is_mix_proc: bool = True):
    """
    将单个 std 草图转化为 unified_std 草图
    :param std_root: [n, 4]
    :param min_pnt: 处理后点数低于该数值的草图剔除
    :param is_mix_proc: 是否反复进行笔划拆分、合并、实现笔划的长度尽量相等
    :return:
    """

    def distance(p1, p2):
        """
        计算两点之间的欧氏距离
        """
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def search_nearest_stroke(stroke_list, end_point, given_idx):
        """
        找到stroke_list中stroke_idx对应的笔划中，end_point点最近的笔划，并返回是合并到起点还是终点
        :param stroke_list:
        :param end_point:
        :param given_idx:
        :return: closest_idx, closet_dist, is_connect_start
        """
        # 计算最短笔划与其他笔划的起始点距离，找到最近的笔划
        closest_idx_start = -1
        min_distance_start = float('inf')

        closest_idx_end = -1
        min_distance_end = float('inf')

        for i in range(len(stroke_list)):
            if i == given_idx:  # 跳过自身
                continue
            dist_start = distance(end_point, stroke_list[i][0])
            if dist_start <= min_distance_start:
                min_distance_start = dist_start
                closest_idx_start = i

            dist_end = distance(end_point, stroke_list[i][-1])
            if dist_end <= min_distance_end:
                min_distance_end = dist_end
                closest_idx_end = i

        if min_distance_start <= min_distance_end:
            if min_distance_start == float('inf'):
                raise ValueError('inf dist occurred')

                # print(stroke_list)
                # stroke_list = np.concatenate(stroke_list, axis=0)
                # np.savetxt('error_stk', stroke_list, delimiter=',')
                # exit(0)

            return closest_idx_start, min_distance_start, True
        else:
            if min_distance_start == float('inf'):
                raise ValueError('inf dist occurred 2')

                # print(stroke_list)
                # stroke_list = np.concatenate(stroke_list, axis=0)
                # np.savetxt('error_stk', stroke_list, delimiter=',')
                # exit(0)

            return closest_idx_end, min_distance_end, False

    def single_merge(stroke_list, dist_gap=0.2):
        """
        将草图中最短的一个笔划合并到其他笔划
        :param stroke_list:
        :param dist_gap: 若某个笔划距其它笔划的最近距离大于该值，不合并
        :return:
        """
        # stroke_list = copy.deepcopy(stk_list)

        # 因距离其它笔画太远不合并的笔划
        stk_cannot_merge = []
        ita_count = 0

        while True:
            if (len(stk_cannot_merge) + len(stroke_list)) > 2 and len(stroke_list) < 2:
                raise ValueError('所有笔划均距其它笔划过远，无法合并，请尝试增加笔划数')

            # 防止迭代次数过多陷入死循环
            if ita_count > 200:
                # print('到达最大迭代次数，不合并该笔划')
                # warnings.warn('cannot merge min stroke')
                raise ValueError('到达最大迭代次数，不合并该笔划')
            else:
                ita_count += 1

            # 找到点数最少的笔划索引
            min_idx = min(range(len(stroke_list)), key=lambda i: len(stroke_list[i]))
            min_stroke = stroke_list[min_idx]

            # if len(min_stroke) == 0:
            #     print(f'file root: {std_root}, 出现最短笔划点数为零')
            #     exit(0)

            min_start = min_stroke[0]  # 最短笔划的起始点
            min_end = min_stroke[-1]  # 最短笔划的起始点

            cidx_st, dist_st, is_ct_st_st = search_nearest_stroke(stroke_list, min_start, min_idx)
            cidx_ed, dist_ed, is_ct_st_ed = search_nearest_stroke(stroke_list, min_end, min_idx)

            # 如果最近的笔画距离其它笔画过远，不合并
            if min(dist_st, dist_ed) > dist_gap:
                stk_cannot_merge.append(min_stroke)
                del stroke_list[min_idx]

            else:
                if dist_st <= dist_ed:
                    closest_idx = cidx_st
                    is_this_start = True
                    is_target_start = is_ct_st_st
                else:
                    closest_idx = cidx_ed
                    is_this_start = False
                    is_target_start = is_ct_st_ed

                # 情形1：起点到起点，target不动，this调转后拼接在前面：
                target_stk = stroke_list[closest_idx]
                if is_this_start and is_target_start:
                    min_stroke = np.flip(min_stroke, axis=0)
                    if distance(min_stroke[-1, :], target_stk[0, :]) < 1e-6:  # 距离过近删除拼接点
                        min_stroke = min_stroke[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形2：起点到终点，this拼接在target后面：
                elif is_this_start and (not is_target_start):
                    if distance(target_stk[-1, :], min_stroke[0, :]) < 1e-6:  # 距离过近删除拼接点
                        target_stk = target_stk[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                # 情形3：终点到起点，this拼接在target前面：
                elif (not is_this_start) and is_target_start:
                    if distance(min_stroke[-1, :], target_stk[0, :]) < 1e-6:  # 距离过近删除拼接点
                        min_stroke = min_stroke[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([min_stroke, target_stk], axis=0)

                # 情形4：终点到终点，target不动，this调转后拼接在后面：
                elif (not is_this_start) and (not is_target_start):
                    min_stroke = np.flip(min_stroke, axis=0)
                    if distance(target_stk[-1, :], min_stroke[0, :]) < 1e-6:  # 距离过近删除拼接点
                        target_stk = target_stk[: -1, :]
                    stroke_list[closest_idx] = np.concatenate([target_stk, min_stroke], axis=0)

                else:
                    warnings.warn('error occurred when stroke merged')
                    raise ValueError('error occurred when stroke merged')

                del stroke_list[min_idx]  # 删除已合并的笔划

                # 加入之前不能合并的距其它笔划过远的笔划
                if len(stk_cannot_merge) != 0:
                    stroke_list = stroke_list + stk_cannot_merge

                return stroke_list

    def single_split(stroke_list):
        """
        将草图中最长的笔画对半分割成两个
        :param stroke_list:
        :return:
        """
        # stroke_list = copy.deepcopy(stk_list)

        # Find the array with the maximum number of rows
        largest_idx = max(range(len(stroke_list)), key=lambda i: stroke_list[i].shape[0])
        largest_array = stroke_list[largest_idx]

        # Split the largest array into two halves
        split_point = largest_array.shape[0] // 2
        first_half = largest_array[:split_point + 1, :]
        second_half = largest_array[split_point:, :]

        # Replace the largest array with the two halves
        del stroke_list[largest_idx]
        stroke_list.append(first_half)
        stroke_list.append(second_half)

        return stroke_list

    # 读取std草图数据
    sketch_data = np.loadtxt(std_root, delimiter=',')
    sketch_data[-1, 2] = global_defs.pen_down

    # 去掉点数过少的笔划
    sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # 去掉笔划上距离过近的点
    sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= min_pnt:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{std_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉过短的笔划
    strokes = sp.stroke_len_filter(strokes, 0.1)

    # 重采样，使得点之间的距离近似相等
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        median_ratio=0.1,
        approx_mode='uni-arclength'
    )

    # 先处理笔划数
    # 大于指定数值，将长度较短的笔划连接到最近的笔划
    n_stk = global_defs.n_stk

    if len(strokes) > n_stk:
        while len(strokes) > n_stk:
            strokes = single_merge(strokes)

    # 小于指定数值，拆分点数较多的笔划
    elif len(strokes) < n_stk:
        while len(strokes) < n_stk:
            strokes = single_split(strokes)

    if len(strokes) != global_defs.n_stk:
        raise ValueError(f'error stroke number: {len(strokes)}')

    if is_mix_proc:
        # 不断拆分、合并笔划，使各笔划点数尽量接近
        n_ita = 0
        while True:
            before_var = sp.stroke_length_var(strokes)

            strokes = single_merge(strokes)
            strokes = single_split(strokes)

            after_var = sp.stroke_length_var(strokes)

            if after_var == before_var or n_ita > 150:
                # print('number of iteration: ', n_ita)
                break

            n_ita += 1

    if len(strokes) != global_defs.n_stk:
        raise ValueError(f'error stroke number final: {len(strokes)}, file: {std_root}')

    # 处理点数
    n_stk_pnt = global_defs.n_stk_pnt
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        min_sample=n_stk_pnt,
        max_sample=n_stk_pnt,
        approx_mode='linear-interp'
    )

    return strokes


def std_unify_batched(source_dir=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt', target_dir=r'D:\document\DeepLearning\DataSet\unified_sketch', is_mix_proc=True):
    """
    将 source_dir 文件夹中的全部 txt 草图转化为 unified_std 草图，并保存到 target_dir
    :param source_dir:
    :param target_dir:
    :return:
    """
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    print('create dirs')
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)

    files_all = get_allfiles(source_dir)

    for c_file in tqdm(files_all, total=len(files_all)):
        try:
            sketch_data = std_unify(c_file, is_mix_proc)
        except:
            warnings.warn(f'error occurred when trans {c_file}, has skipped!')
            continue

        if sketch_data is not None:
            try:
                sketch_data = np.concatenate(sketch_data, axis=0)
                sketch_data = sketch_data[:, :2]

                transed_npnts = len(sketch_data)
                if transed_npnts == global_defs.n_stk * global_defs.n_stk_pnt:

                    target_save_path = c_file.replace(source_dir, target_dir)
                    np.savetxt(target_save_path, sketch_data, delimiter=',')
                else:
                    warnings.warn(f'current point number is {transed_npnts}, skip file trans: {c_file}')
            except:
                warnings.warn(f'error occurred when trans {c_file}, has skipped!')


def create_tree_like(source_dir, target_dir):
    """
    在target_dir下创建与source_dir相同的目录层级
    :param source_dir:
    :param target_dir:
    :return:
    """
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)


def cls_distribute(source_dir, target_dir, test_rate=0.2):
    """
    定位文件的路径如下：
    source_dir
    ├─ Bushes
    │   ├─0.obj
    │   ├─1.obj
    │   ...
    │
    ├─ Clamps
    │   ├─0.obj
    │   ├─1.obj
    │   ...
    │
    ...

    """
    os.makedirs(target_dir, exist_ok=True)
    # 清空target_dir
    print('clear dir: ', target_dir)
    shutil.rmtree(target_dir)

    # 在target_dir中创建与source_dir相同的目录层级
    print('create dirs')
    create_tree_like(source_dir, os.path.join(target_dir, 'train'))
    create_tree_like(source_dir, os.path.join(target_dir, 'test'))

    # 复制文件
    # 获取全部类别
    classes_all = get_subdirs(source_dir)
    for c_class in tqdm(classes_all, total=len(classes_all)):

        c_class_dir = os.path.join(source_dir, c_class)
        c_files = get_allfiles(c_class_dir)

        n_files_all = len(c_files)
        n_files_test = math.ceil(n_files_all * test_rate)

        c_target_train = os.path.join(target_dir, 'train', c_class)
        c_target_test = os.path.join(target_dir, 'test', c_class)
        for i in range(n_files_all):
            base_name = os.path.basename(c_files[i])
            if i < n_files_test:
                shutil.copy(c_files[i], os.path.join(c_target_test, base_name))
            else:
                shutil.copy(c_files[i], os.path.join(c_target_train, base_name))


def quickdraw_download(target_dir):
    # 先提取全部类别
    class_file = r'C:\Users\ChengXi\Desktop\quickdraw.txt'
    class_all = []

    with open(class_file, 'r') as f:
        for c_line in f.readlines():
            c_line = c_line.strip()
            if 'full' in c_line:
                class_all.append(c_line.split('/')[-2])

    # print(class_all)
    # print(len(class_all))

    os.makedirs(target_dir, exist_ok=True)

    for category in tqdm(class_all, total=len(class_all)):
        url = f"https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category}"
        response = requests.get(url)
        with open(os.path.join(target_dir, category), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {category}")


def pen_updown_alt(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # 去除换行符并分割字段
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue  # 跳过格式不正确的行（可选）

            a, b, c, d = parts

            # 转换c字段：16→0，17→1
            new_c = '0' if c == '16' else '1'

            # 构造新行并写入（自动添加换行符）
            f_out.write(f"{a},{b},{new_c}\n")

def pen_updown_alt_batched(source_dir, target_dir, up_before=16, down_before=17, up_after=0, down_after=1):
    """
    修改源文件夹里的草图的up down指令
    :param source_dir:
    :param target_dir:
    :param up_before:
    :param down_before:
    :param up_after:
    :param down_after:
    :return:
    """
    # 先在target_dir创建与source_dir相同的目录层级
    print('clear dir:', target_dir)
    os.makedirs(target_dir, exist_ok=True)
    shutil.rmtree(target_dir)
    create_tree_like(source_dir, target_dir)

    files_all = get_allfiles(source_dir)

    for c_file in tqdm(files_all, total=len(files_all)):
        target_file = c_file.replace(source_dir, target_dir)
        pen_updown_alt(c_file, target_file)


def equal_dist_resample(stroke: np.ndarray, resp_dist: float) -> np.ndarray:
    """
    等距采样
    :param stroke:
    :param resp_dist:
    :return:
    """
    # 将 ndarray 转换为列表，每个元素为长度为2的 ndarray
    stroke = [stroke[i, :2] for i in range(stroke.shape[0])]
    resampled = [stroke[0]]  # 添加第一个点
    accum_dist = 0.0  # 上一个采样点与当前点累积的距离
    i = 1

    while i < len(stroke):
        c_dist = np.linalg.norm(stroke[i] - stroke[i - 1])

        if accum_dist + c_dist >= resp_dist:
            interp_pnt = stroke[i - 1] + (resp_dist - accum_dist) * (stroke[i] - stroke[i - 1])
            resampled.append(interp_pnt)
            stroke.insert(i, interp_pnt)
            accum_dist = 0.0

        else:
            accum_dist += c_dist

        i += 1

    # 如果最后一个点与最后一个采样点之间的距离小于 0.5 * resp_dist，可额外加入最后一个点
    # if np.linalg.norm(stroke_list[-1] - resampled[-1]) < 0.5 * resp_dist:
    #     resampled.append(stroke_list[-1])

    # 将采样后的点列表转换为 n×2 的 ndarray
    return np.vstack(resampled)


def merge_point_group(stroke: np.ndarray, splits_raw: list, dist_thres: float) -> list:
    """
    将 stroke（n×2 的 ndarray）中的分割点 splits_raw 根据距离阈值 dist_thres 进行合并，
    返回每个分割组中间的索引列表。
    """
    if len(splits_raw) == 0 or len(splits_raw) >= stroke.shape[0]:
        return []

    if len(splits_raw) == 1:
        return splits_raw

    # 将索引转化为索引与起点到该索引点距离的元组，用于集群表示
    idx_and_arclen = []
    for c_split in splits_raw:
        arc_len = sp.stroke_length(stroke[:c_split + 1, :])

        idx_and_arclen.append((c_split, arc_len))

    groups = []       # 最终的分组列表
    current_group = [idx_and_arclen[0]]  # 当前分组，初始放入第一个元组

    # 遍历数组，从第二个元素开始
    for tup in idx_and_arclen[1:]:
        # 如果当前元组的 dist 与当前分组最后一个元组的 dist 差值小于阈值，
        # 则认为它们属于同一组
        if tup[1] - current_group[-1][1] < dist_thres:
            current_group.append(tup)
        else:
            groups.append(current_group)
            current_group = [tup]  # 开启新一组

    # 添加最后一个分组
    groups.append(current_group)

    # 取各组的中间值
    merged_idx = []
    for c_group in groups:
        if len(c_group) == 1:
            merged_idx.append(c_group[0][0])
        else:
            idx_left = c_group[0][0]
            idx_right = c_group[-1][0]

            merged_idx.append((idx_left + idx_right) // 2)

    return merged_idx

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # c_overall_idx = 0
    # n_splits = len(splits_raw)
    #
    # # 第一个分割点作为初始组的第一个点
    # # 该数组存放分割组最左边的点，后续需要处理，将其处理成分割组中间的点
    # # split_idx = [splits_raw[0]]
    #
    # # 存储每个分割组中第一个和最后一个索引的元组
    # split_idx_start_end = [splits_raw[0]]
    # c_left = stroke[splits_raw[0]]
    #
    # for i in range(1, n_splits):
    #     # 计算 stroke 中两个分割点之间的欧氏距离
    #     # c_left = stroke[splits_raw[c_overall_idx]]
    #     c_split = stroke[splits_raw[i]]
    #
    #     if np.linalg.norm(c_left - c_split) > dist_thres:
    #         split_idx.append(splits_raw[i])
    #         c_overall_idx = i
    #         # 使用当前组中前一个采样点和上一个 split 点作为起始和结束索引
    #         # 有问题，如果某个分割组只有一个点的话
    #         split_idx_start_end.append((split_idx[-2], splits_raw[i - 1]))
    #
    # # 如果最后两个分割点距离小于等于阈值，则把最后一个点加入对应分割组
    # if n_splits > 2 and np.linalg.norm(stroke[splits_raw[-1]] - stroke[splits_raw[-2]]) <= dist_thres:
    #     split_idx_start_end.append((split_idx[-1], splits_raw[-1]))
    #
    # # 清空 split_idx 并用每个分割组中间的索引替代（整数除法，取整）
    # split_idx = []
    # for first, second in split_idx_start_end:
    #     split_idx.append((first + second) // 2)
    #
    # return split_idx


def short_straw_split(stroke: np.ndarray, resp_dist: float, filter_dist: float, thres: float, window_width: int, is_show_status=False):
    """
    利用short straw进行角点分割，使用前必须将草图进行归一化至质心(0, 0)，范围[-1, 1]^2
    :param stroke: 单个笔划 [n, 2]
    :param resp_dist: 重采样间隔 [0, 1]
    :return:
    """
    # 辅助检测草图是否被归一化过
    # max_val = np.max(stroke)
    # min_val = np.min(stroke)
    assert 1 - np.max(stroke) >= -1e-5 and np.min(stroke) + 1 >= -1e-5
    assert stroke.shape[1] == 2

    if is_show_status:
        all_straw = []

    splited_stk = []

    # Step 1: 重采样
    resample_stk = sp.uni_arclength_resample_strict_single(stroke, resp_dist)

    # 如果重采样后的点数小于 window_width 的3倍，则返回原数组
    if len(resample_stk) < 3 * window_width:
        splited_stk.append(stroke)
        return splited_stk

    # Step 2: 计算 straw 值
    straw_and_idx = []
    straw_base = 0.0
    n_resample = len(resample_stk)
    half_window = window_width // 2

    for i in range(half_window, n_resample - half_window):
        window_left = i - half_window
        window_right = i + half_window

        pnt_left = resample_stk[window_left]
        pnt_right = resample_stk[window_right]
        c_straw = np.linalg.norm(pnt_left - pnt_right)
        straw_and_idx.append((c_straw, i))

        if is_show_status:
            all_straw.append(c_straw)

        if c_straw > straw_base:
            straw_base = c_straw

    # Step 3: 根据 straw 阈值确定角点
    straw_thres = straw_base * thres
    m_corners_idx = [idx for (straw, idx) in straw_and_idx if straw < straw_thres]

    # for idx, c_val in enumerate(straw_and_idx):
    #     if c_val[0] < straw_thres:
    #         asasas = 0

    # Step 4: 合并过于接近的角点
    m_corners_idx = merge_point_group(resample_stk, m_corners_idx, filter_dist)

    # Step 5: 根据角点分割重采样后的笔画
    splited_stk = np.split(resample_stk, m_corners_idx, axis=0)

    if is_show_status:
        fig, axs = plt.subplots(1, 2)

        axs[0].plot(stroke[:, 0], -stroke[:, 1])

        x = range(len(all_straw))
        axs[1].plot(x, all_straw)
        axs[1].plot([x[0], x[-1]], [straw_thres, straw_thres])

        plt.show()

    return splited_stk


def sketch_short_straw_split(sketch, resp_dist: float = 0.01, filter_dist: float = 0.01, thres: float = 0.9, window_width: int = 3, split_length: float = 0.2, is_print_split_status=False):
    splited_sketch = []

    for c_stk in sketch:
        if sp.stroke_length(c_stk) > split_length:
            splited_sketch += short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, is_print_split_status)

    return splited_sketch


def short_straw_split_sketch(sketch_root: str, resp_dist: float = 0.01, filter_dist: float = 0.1, thres: float = 0.9, window_width: int = 3,
                             pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, is_show_status=True) -> list:
    """
    利用short straw对彩图进行角点分割
    :param sketch_root:
    :param resp_dist:
    :param filter_dist:
    :param thres:
    :param window_width:
    :param is_show_status:
    :return:
    """

    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割草图
    sketch_data = sp.sketch_split(sketch_data, pen_up, pen_down)

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 5)

    # 去掉笔划上距离过近的点
    # sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.03)

    # 去掉长度过短的笔划
    # sketch_data = sp.stroke_len_filter(sketch_data, 0.07)

    tmp_vis_sketch_list(sketch_data)

    if n_sketch_pnt(sketch_data) <= 20:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
        return []

    # 分割笔划
    strokes_splited = []
    for c_stk in sketch_data:
        strokes_splited += short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, False)

    tmp_vis_sketch_list(strokes_splited)

    # 去掉点数过少的笔划
    # strokes_splited = sp.stk_pnt_num_filter(strokes_splited, 16)

    # 去掉笔划上距离过近的点
    # strokes_splited = sp.near_pnt_dist_filter(strokes_splited, 0.03)

    # 去掉长度过短的笔划
    # strokes_splited = sp.stroke_len_filter(strokes_splited, 0.07)

    # 仅保留点数最多的前 global_def.n_stk 个笔划
    strokes_splited = sp.stk_number_filter(strokes_splited, global_defs.n_stk)

    # 每个笔划中的点数仅保留前 global_def.n_pnt 个
    strokes_splited = sp.stk_pnt_filter(strokes_splited, global_defs.n_stk_pnt)

    if is_show_status:
        for s in strokes_splited:
            plt.plot(s[:, 0], -s[:, 1])
            plt.scatter(s[:, 0], -s[:, 1])
        plt.show()

    return strokes_splited


def pre_process(sketch_root: str, resp_dist: float = 0.01, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
    """
    TODO: 需考虑太长笔划和太短笔划不能放在一起的问题，必须进行分割
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :return:
    """
    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割笔划
    sketch_data = sp.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉相邻过近的点
    # -----------------需要先归一化才可使用，不然单位不统一
    sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.001)

    # 重采样
    # sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # 角点分割
    sketch_data = sketch_short_straw_split(sketch_data, resp_dist, is_print_split_status=False)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # tmp_vis_sketch_list(sketch_data, True)

    # 去掉无效笔划
    # sketch_data = sp.valid_stk_filter(sketch_data)

    # 长笔划分割
    sketch_data = sp.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    sketch_data = sp.stk_pnt_num_filter(sketch_data, 8)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 使所有笔划的点数均为2的整数倍
    sketch_data = sp.stk_pnt_double_filter(sketch_data)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
    sketch_data = sp.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
    sketch_data = sp.stk_num_minimal_filter(sketch_data, 4)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
    sketch_data = sp.stk_number_filter(sketch_data, global_defs.n_stk)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # tmp_vis_sketch_list(sketch_data)
    # tmp_vis_sketch_list(sketch_data, True)

    return sketch_data


def pre_process_seg_only(sketch_root: str, resp_dist: float = 0.03, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down) -> list:
    """
    :param sketch_root:
    :param resp_dist:
    :param pen_up:
    :param pen_down:
    :return:
    """
    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割笔划
    sketch_data = sp.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉相邻过近的点
    # -----------------需要先归一化才可使用，不然单位不统一
    sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.01)

    # 去掉长度过短的笔划
    sketch_data = sp.stroke_len_filter(sketch_data, 0.1)

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # 角点分割
    # sketch_data = sketch_short_straw_split(sketch_data, resp_dist, is_print_split_status=False)

    # tmp_vis_sketch_list(sketch_data, True)

    # 去掉无效笔划
    # sketch_data = sp.valid_stk_filter(sketch_data)

    # 长笔划分割
    sketch_data = sp.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    sketch_data = sp.stk_pnt_num_filter(sketch_data, 8)

    # 使所有笔划的点数均为2的整数倍
    sketch_data = sp.stk_pnt_double_filter(sketch_data)

    # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
    sketch_data = sp.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
    # sketch_data = sp.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
    sketch_data = sp.stk_number_filter(sketch_data, global_defs.n_stk)

    # tmp_vis_sketch_list(sketch_data)
    # tmp_vis_sketch_list(sketch_data, True)

    return sketch_data


def test_resample():
    # sketch_root = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\motorbike\10722.txt'
    sketch_root = sketch_test

    # 读取草图数据
    sketch_data = np.loadtxt(sketch_root, delimiter=',')
    # sketch_data[-1, 2] = global_defs.pen_down

    # 去掉点数过少的笔划
    sketch_data = sp.stk_pnt_num_filter(sketch_data, 5)

    # 去掉笔划上距离过近的点
    sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= 25:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉长度过短的笔划
    strokes = sp.stroke_len_filter(strokes, 0.07)

    # 重采样
    strokes_resampled = sp.uni_arclength_resample_strict(strokes, 0.1)

    for s in strokes_resampled:
        plt.plot(s[:, 0], -s[:, 1])
        plt.scatter(s[:, 0], -s[:, 1])

    plt.show()


if __name__ == '__main__':
    # svg_to_txt_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin\sketches', r'D:\document\DeepLearning\DataSet\TU_Berlin_txt')
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin_txt', r'D:\document\DeepLearning\DataSet\TU_Berlin_std')
    # cls_distribute(r'D:\document\DeepLearning\DataSet\TU_Berlin_std', r'D:\document\DeepLearning\DataSet\TU_Berlin_std_cls')


    # --------------------- 草图标准化
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch_cad\sketch_txt', rf'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')
    # object_name = 'cup'
    # std_unify_batched(
    #     rf'D:/document/DeepLearning/DataSet/sketch_from_quickdraw/{object_name}',
    #     rf'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/{object_name}_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}'
    # )

    # std_unify_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt', rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # cls_distribute(rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',
    #                rf'D:\document\DeepLearning\DataSet\TU_Berlin_cls_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\cup\5126.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\motorbike\10722.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt\camera\3285.txt'
    # sketch_test = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt\train\Rivet\6dcc4bc1223014b26abb72b3dee939a8_1.txt'
    #
    # short_straw_split_sketch(sketch_test, is_show_status=False)

    # cls_distribute(r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt', r'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt_cls')

    # test()
    # test_resample()

    # quickdraw_download(r'D:\document\DeepLearning\DataSet\quickdraw_all')

    # pen_updown_alt_batched(r'D:\document\DeepLearning\DataSet\sketch_cad\sketch_txt', r'D:\document\DeepLearning\DataSet\sketch_cad\new')

    # 'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls\\train\\armchair\\530.txt'
    # exsketch = r'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls\\train\\armchair\\530.txt'
    # asketch = pre_process_seg_only(exsketch)
    # tmp_vis_sketch_list(asketch, True)

    # all_sketches = get_allfiles(r'D:\\document\\DeepLearning\\DataSet\\TU_Berlin\\TU_Berlin_txt_cls')
    all_sketches = get_allfiles(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt')

    for c_skh in all_sketches:
        asketch = pre_process_seg_only(c_skh)
        tmp_vis_sketch_list(asketch, True)


    pass





