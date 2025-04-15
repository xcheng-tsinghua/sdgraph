import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from data_utils.sketch_utils import get_allfiles


def stroke_points_statis(root=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt', pen_up=16, pen_down=17, decrease=0.95, is_read_data=False):
    """
    统计每个笔划内的点数分布
    sketch file is saved as txt
    :param root:
    :param decrease:
    :param is_read_data:
    :return:
    """
    if not is_read_data:
        files_all = get_allfiles(root, 'txt')

        stk_pnt_statis = {}

        for c_sketch_root in tqdm(files_all, total=len(files_all)):
            sketch_data = np.loadtxt(c_sketch_root, delimiter=',')
            sketch_data[-1, 2] = pen_down
            strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

            for c_stk in strokes:
                c_stk_pnts = c_stk.shape[0]

                if c_stk_pnts not in stk_pnt_statis.keys():
                    stk_pnt_statis[c_stk_pnts] = 1
                else:
                    stk_pnt_statis[c_stk_pnts] += 1

        print('每个笔划内的点数分布:', stk_pnt_statis)

    else:
        with open('../bks/machinery_sketch_statistic.json', 'r', encoding='utf-8') as file:
            stk_pnt_statis = json.load(file)[0]

        if decrease < 1:
            stk_pnt_statis = ex_some_dict_items(stk_pnt_statis, decrease)

        stk_pnt_statis = {30: 2895, 3: 15295, 10: 6040, 20: 3196, 6: 11075, 28: 3409, 7: 9761, 16: 2476, 5: 14635, 1: 2561, 2: 15979, 19: 2911, 39: 899, 9: 7552, 13: 3166, 29: 3205, 43: 462, 24: 3696, 23: 3833, 25: 3913, 4: 13210, 8: 8603, 21: 3380, 31: 2755, 15: 2485, 22: 3607, 26: 3630, 27: 3459, 12: 3697, 11: 4891, 18: 2714, 17: 2611, 34: 1905, 37: 1270, 36: 1440, 35: 1607, 47: 241, 42: 599, 32: 2478, 40: 820, 41: 673, 14: 2826, 52: 102, 33: 2169, 48: 210, 38: 1133, 53: 86, 50: 136, 49: 192, 46: 286, 45: 360, 57: 46, 56: 40, 51: 140, 44: 406, 62: 17, 54: 66, 59: 30, 61: 22, 58: 42, 60: 25, 55: 65}

        ex_matlab(stk_pnt_statis)

        plt.clf()
        plt.figure(figsize=(18, 5))
        plt.bar(list(stk_pnt_statis.keys()), list(stk_pnt_statis.values()))
        plt.title('number of stroke points statistic')
        plt.xlabel('number of stroke points')
        plt.ylabel('number of strokes')
        plt.show()


def sketch_points_statis(root=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt', decrease=0.95, is_read_data=False):
    """
    统计每个草图内的点数分布
    sketch file is saved as txt
    :param root:
    :param decrease:
    :param is_read_data:
    :return:
    """
    if not is_read_data:
        files_all = get_allfiles(root, 'txt')

        stk_pnt_statis = {}

        for c_sketch_root in tqdm(files_all, total=len(files_all)):
            sketch_data = np.loadtxt(c_sketch_root, delimiter=',')

            c_sketch_pnts = sketch_data.shape[0]

            if c_sketch_pnts not in stk_pnt_statis.keys():
                stk_pnt_statis[c_sketch_pnts] = 1
            else:
                stk_pnt_statis[c_sketch_pnts] += 1

        print('每个草图内的点数分布:', stk_pnt_statis)

    else:
        with open('../bks/machinery_sketch_statistic.json', 'r', encoding='utf-8') as file:
            stk_pnt_statis = json.load(file)[1]

        if decrease < 1:
            stk_pnt_statis = ex_some_dict_items(stk_pnt_statis, decrease)

        stk_pnt_statis = {43: 2175, 26: 1946, 35: 2810, 22: 1047, 30: 2574, 31: 2725, 61: 382, 48: 1414, 27: 2172, 28: 2300, 32: 2809, 46: 1713, 37: 2700, 24: 1472, 44: 2014, 38: 2735, 34: 2818, 42: 2216, 39: 2627, 36: 2854, 25: 1683, 49: 1314, 41: 2343, 29: 2488, 54: 852, 40: 2406, 53: 917, 33: 2891, 56: 657, 59: 435, 45: 1872, 50: 1168, 23: 1244, 52: 1024, 55: 711, 47: 1536, 51: 1119, 62: 327, 58: 556, 60: 387, 57: 567}
        ex_matlab(stk_pnt_statis)

        plt.clf()
        plt.figure(figsize=(18, 5))
        plt.bar(list(stk_pnt_statis.keys()), list(stk_pnt_statis.values()))
        plt.title('number of sketch points statistic')
        plt.xlabel('number of sketch points')
        plt.ylabel('number of sketches')
        plt.show()


def stroke_statis(root=r'D:\document\DeepLearning\DataSet\sketch\sketch_txt', pen_up=16, pen_down=17, decrease=0.95, is_read_data=False):
    """
    统计每草图个内的笔划数
    sketch file is saved as txt
    :param root:
    :param decrease:
    :param is_read_data:
    :return:
    """
    if not is_read_data:
        files_all = get_allfiles(root, 'txt')

        stk_pnt_statis = {}

        for c_sketch_root in tqdm(files_all, total=len(files_all)):
            sketch_data = np.loadtxt(c_sketch_root, delimiter=',')
            sketch_data[-1, 2] = pen_down
            strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

            c_stk_num = len(strokes)

            if c_stk_num not in stk_pnt_statis.keys():
                stk_pnt_statis[c_stk_num] = 1
            else:
                stk_pnt_statis[c_stk_num] += 1

        print('每草图个内的笔划数:', stk_pnt_statis)

    else:
        with open('../bks/machinery_sketch_statistic.json', 'r', encoding='utf-8') as file:
            stk_pnt_statis = json.load(file)[2]

        if decrease < 1:
            stk_pnt_statis = ex_some_dict_items(stk_pnt_statis, decrease)

        stk_pnt_statis = {3: 23889, 2: 23649, 4: 9354, 1: 8472, 5: 2907, 7: 377, 6: 967, 11: 36, 10: 46, 8: 173, 9: 85, 12: 21, 13: 9, 16: 3, 14: 6, 15: 2, 17: 4}
        ex_matlab(stk_pnt_statis)

        plt.clf()
        plt.figure(figsize=(18, 5))
        plt.bar(list(stk_pnt_statis.keys()), list(stk_pnt_statis.values()))
        plt.title('stroke number of statistic')
        plt.xlabel('number of strokes')
        plt.ylabel('number of sketches')
        plt.show()


def ex_some_dict_items(adict: dict, perc: float) -> dict:
    new_dict = {}

    total = len(adict)
    n_ex = total * perc

    for i, key in enumerate(adict.keys()):
        if i < n_ex:
            new_dict[key] = adict[key]

    return new_dict


def ex_matlab(adict: dict):
    """
    将python的字典打印出来，以方便复制进MATLAB可视化
    :param adict:
    :return:
    """
    keys = list(adict.keys())
    values = list(adict.values())

    # 格式化为 MATLAB 的代码
    keys_str = "keys = [" + " ".join(map(str, keys)) + "];"
    values_str = "values = [" + " ".join(map(str, values)) + "];"

    # 打印 MATLAB 代码
    print(keys_str)
    print(values_str)


if __name__ == '__main__':
    quick_draw_root = r'D:\document\DeepLearning\DataSet\sketch_from_quickdraw\apple\train'

    stroke_points_statis(quick_draw_root, 0, 1, is_read_data=True)
    sketch_points_statis(quick_draw_root, is_read_data=True)
    stroke_statis(quick_draw_root, 0, 1, is_read_data=True)


