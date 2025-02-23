import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from svgpathtools import svg2paths2
import warnings
import shutil
import math
from tqdm import tqdm

import global_defs
import encoders.spline as sp


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
            if file.split('.')[-1] == suffix:
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
    将 source_dir 文件夹中的全部 std 草图转化为 unified_std 草图，并保存到 target_dir
    :param source_dir:
    :param target_dir:
    :return:
    """
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
    for c_class in classes_all:
        print(c_class)

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


if __name__ == '__main__':
    # svg_to_txt_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin\sketches', r'D:\document\DeepLearning\DataSet\TU_Berlin_txt')
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\TU_Berlin_txt', r'D:\document\DeepLearning\DataSet\TU_Berlin_std')
    # cls_distribute(r'D:\document\DeepLearning\DataSet\TU_Berlin_std', r'D:\document\DeepLearning\DataSet\TU_Berlin_std_cls')


    # --------------------- 草图标准化
    # std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch\sketch_txt',r'D:\document\DeepLearning\DataSet\unified_sketch')
    std_unify_batched(r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple', f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')

    pass





