import os
from PIL import Image
from pytorch_fid import fid_score
import tempfile
import shutil
import numpy as np
import torch


def resize_images(input_dir, output_dir, size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.BILINEAR)
            img.save(os.path.join(output_dir, filename))
        except:
            print(f'error file : {img_path}')
            exit(1)


def resize_images_no_color(input_dir, output_dir, size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.BILINEAR)

            # ---- 新增：彩色图转黑白二值 ----
            arr = np.array(img)

            # 定义“白色”判断阈值
            white_thresh = 240

            # 判断是否为白色：三个通道都 > 240
            is_white = np.all(arr > white_thresh, axis=-1)

            # 创建一个全白图
            bw = np.ones_like(arr) * 255

            # 不是白色的部分全部变成黑色
            bw[~is_white] = 0

            img = Image.fromarray(bw.astype('uint8'))
            # ---- 新增功能结束 ----

            img.save(os.path.join(output_dir, filename))
        except:
            print(f'error file : {img_path}')
            exit(1)


def compute_fid(real_dir, fake_dir):
    # 创建两个临时文件夹用于保存resize后的图像
    tmp_real = tempfile.mkdtemp()
    tmp_fake = tempfile.mkdtemp()

    try:
        # print("Resizing images...")
        resize_images_no_color(real_dir, tmp_real)
        resize_images_no_color(fake_dir, tmp_fake)

        # print("Computing FID...")
        fid_value = fid_score.calculate_fid_given_paths([tmp_real, tmp_fake], batch_size=50, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)

        return fid_value
        # print(f'FID: {fid_value:.4f}')

    finally:
        shutil.rmtree(tmp_real)
        shutil.rmtree(tmp_fake)


if __name__ == "__main__":
    cats = ['apple', 'moon', 'book', 'shark', 'angel', 'bicycle']

    for c_cat in cats:

        # 设置你的目录路径
        real_images_path = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\{c_cat}'

        fake_images_path_sdgraph = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sdgraph\{c_cat}_11_16'
        fake_images_path_sketchknitter = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sketchknitter\{c_cat}'
        fake_images_path_sketchrnn = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sketchrnn\{c_cat}'

        DC_gra2seq = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\DC-gra2seq\{c_cat}'
        SketchHealer = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\SketchHealer\{c_cat}'
        sketch_lattice = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sketch-lattice\{c_cat}'
        SP_gra2seq = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\SP-gra2seq\{c_cat}'

        fid_1 = compute_fid(real_images_path, fake_images_path_sdgraph)
        print(c_cat, f'sdgraph: {fid_1}')

        fid_2 = compute_fid(real_images_path, fake_images_path_sketchknitter)
        print(c_cat, f'sketchknitter: {fid_2}')

        fid_3 = compute_fid(real_images_path, fake_images_path_sketchrnn)
        print(c_cat, f'sketchrnn: {fid_3}')

        fid_4 = compute_fid(real_images_path, DC_gra2seq)
        print(c_cat, f'DC_gra2seq: {fid_4}')

        fid_5 = compute_fid(real_images_path, SketchHealer)
        print(c_cat, f'SketchHealer: {fid_5}')

        fid_6 = compute_fid(real_images_path, sketch_lattice)
        print(c_cat, f'sketch_lattice: {fid_6}')

        fid_7 = compute_fid(real_images_path, SP_gra2seq)
        print(c_cat, f'SP_gra2seq: {fid_7}')





