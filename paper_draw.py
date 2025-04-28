import numpy as np

from data_utils import sketch_utils
from data_utils import vis
import random







if __name__ == '__main__':

    data_root = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all'

    files_all = sketch_utils.get_allfiles(data_root)
    random.shuffle(files_all)

    for c_file in files_all:
        vis.vis_sketch_orig(c_file, title=c_file, show_dot=True, dot_gap=5)

        sketch_utils.std_to_tensor_img(np.loadtxt(c_file, delimiter=','))



    # asasad = r'E:\document\iDesignCAD-Assistant\iDesignCAD\data.txt'
    # vis.vis_sketch_orig(asasad, title=asasad)






    pass












