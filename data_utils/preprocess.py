import torch
import numpy as np
import warnings
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

import global_defs
from data_utils import filter as ft
from encoders import spline as sp
from data_utils import data_utils as du
from data_utils import vis


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
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 4)

    # 去掉笔划上距离过近的点
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= min_pnt:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{std_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉过短的笔划
    strokes = ft.stroke_len_filter(strokes, 0.1)

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
            before_var = du.stroke_length_var(strokes)

            strokes = single_merge(strokes)
            strokes = single_split(strokes)

            after_var = du.stroke_length_var(strokes)

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

    files_all = du.get_allfiles(source_dir)

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
    sketch_data = du.sketch_std(sketch_data)

    # 分割草图
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 5)

    # 去掉笔划上距离过近的点
    # sketch_data = sp.near_pnt_dist_filter(sketch_data, 0.03)

    # 去掉长度过短的笔划
    # sketch_data = sp.stroke_len_filter(sketch_data, 0.07)

    vis.vis_sketch_list(sketch_data)

    if du.n_sketch_pnt(sketch_data) <= 20:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
        return []

    # 分割笔划
    strokes_splited = []
    for c_stk in sketch_data:
        strokes_splited += du.short_straw_split(c_stk[:, :2], resp_dist, filter_dist, thres, window_width, False)

    vis.vis_sketch_list(strokes_splited)

    # 去掉点数过少的笔划
    # strokes_splited = sp.stk_pnt_num_filter(strokes_splited, 16)

    # 去掉笔划上距离过近的点
    # strokes_splited = sp.near_pnt_dist_filter(strokes_splited, 0.03)

    # 去掉长度过短的笔划
    # strokes_splited = sp.stroke_len_filter(strokes_splited, 0.07)

    # 仅保留点数最多的前 global_def.n_stk 个笔划
    strokes_splited = ft.stk_number_filter(strokes_splited, global_defs.n_stk)

    # 每个笔划中的点数仅保留前 global_def.n_pnt 个
    strokes_splited = ft.stk_pnt_filter(strokes_splited, global_defs.n_stk_pnt)

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
    sketch_data = du.sketch_std(sketch_data)

    # 分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉相邻过近的点
    # -----------------需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.001)

    # 重采样
    # sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # 角点分割
    sketch_data = du.sketch_short_straw_split(sketch_data, resp_dist, is_print_split_status=False)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # tmp_vis_sketch_list(sketch_data, True)

    # 去掉无效笔划
    # sketch_data = sp.valid_stk_filter(sketch_data)

    # 长笔划分割
    sketch_data = ft.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 8)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 使所有笔划的点数均为2的整数倍
    sketch_data = ft.stk_pnt_double_filter(sketch_data)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
    sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
    sketch_data = ft.stk_num_minimal_filter(sketch_data, 4)

    if len(sketch_data) == 0:
        print(f'occurred zero sketch: {sketch_root}')
        return [torch.zeros(global_defs.n_stk_pnt, 2, dtype=torch.float).numpy()]

    # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
    sketch_data = ft.stk_number_filter(sketch_data, global_defs.n_stk)

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
    sketch_data = du.sketch_std(sketch_data)

    # 分割笔划
    sketch_data = du.sketch_split(sketch_data, pen_up, pen_down)

    # 去掉相邻过近的点
    # -----------------需要先归一化才可使用，不然单位不统一
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.01)

    # 去掉长度过短的笔划
    sketch_data = ft.stroke_len_filter(sketch_data, 0.1)

    # 重采样
    sketch_data = sp.uni_arclength_resample_strict(sketch_data, resp_dist)

    # 角点分割
    # sketch_data = sketch_short_straw_split(sketch_data, resp_dist, is_print_split_status=False)

    # tmp_vis_sketch_list(sketch_data, True)

    # 去掉无效笔划
    # sketch_data = sp.valid_stk_filter(sketch_data)

    # 长笔划分割
    sketch_data = ft.stk_n_pnt_maximum_filter(sketch_data, global_defs.n_stk_pnt)

    # tmp_vis_sketch_list(sketch_data)

    # 去掉点数过少的笔划
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 8)

    # 使所有笔划的点数均为2的整数倍
    sketch_data = ft.stk_pnt_double_filter(sketch_data)

    # 每个笔划中的点数过多时，仅保留前 global_def.n_pnt 个
    sketch_data = ft.stk_pnt_filter(sketch_data, global_defs.n_stk_pnt)

    # 有效笔划数必须大于指定值，否则图节点之间的联系将不复存在
    # sketch_data = sp.stk_num_minimal_filter(sketch_data, 4)

    # 有效笔划数大于上限时，仅保留点数最多的前 global_def.n_stk 个笔划
    sketch_data = ft.stk_number_filter(sketch_data, global_defs.n_stk)

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
    sketch_data = ft.stk_pnt_num_filter(sketch_data, 5)

    # 去掉笔划上距离过近的点
    sketch_data = ft.near_pnt_dist_filter(sketch_data, 0.05)

    if len(sketch_data) <= 25:
        warnings.warn(f'筛选后的草图点数太少，不处理该草图：{sketch_root}！点数：{len(sketch_data)}')
        return None

    # 移动草图质心与大小
    sketch_data = du.sketch_std(sketch_data)

    # 分割草图笔划
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == global_defs.pen_up)[0] + 1)

    # 去掉长度过短的笔划
    strokes = ft.stroke_len_filter(strokes, 0.07)

    # 重采样
    strokes_resampled = sp.uni_arclength_resample_strict(strokes, 0.1)

    for s in strokes_resampled:
        plt.plot(s[:, 0], -s[:, 1])
        plt.scatter(s[:, 0], -s[:, 1])

    plt.show()
    pass



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
    all_sketches = du.get_allfiles(rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt')

    for c_skh in all_sketches:
        asketch = pre_process_seg_only(c_skh)
        vis.vis_sketch_list(asketch, True)


    pass

