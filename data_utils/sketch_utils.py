import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path


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

