import numpy
import numpy as np

from data_utils import sketch_utils
from data_utils import vis
from data_utils import preprocess
import random







if __name__ == '__main__':

    # # data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all'
    data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Nut'

    files_all = sketch_utils.get_allfiles(data_root)
    random.shuffle(files_all)

    for c_file in files_all:
        print(c_file)
        vis.vis_sketch_orig(c_file, title=c_file, show_dot=True, dot_gap=1)
        preprocess.preprocess_orig(c_file, is_show_status=True)
        # sketch_utils.std_to_tensor_img(np.loadtxt(c_file, delimiter=','))



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


    pass












