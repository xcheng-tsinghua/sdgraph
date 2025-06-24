import os
from PIL import Image
from pytorch_fid import fid_score
import tempfile
import shutil


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


def compute_fid(real_dir, fake_dir):
    # 创建两个临时文件夹用于保存resize后的图像
    tmp_real = tempfile.mkdtemp()
    tmp_fake = tempfile.mkdtemp()

    try:
        # print("Resizing images...")
        resize_images(real_dir, tmp_real)
        resize_images(fake_dir, tmp_fake)

        # print("Computing FID...")
        fid_value = fid_score.calculate_fid_given_paths([tmp_real, tmp_fake], batch_size=50, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)

        return fid_value
        # print(f'FID: {fid_value:.4f}')

    finally:
        shutil.rmtree(tmp_real)
        shutil.rmtree(tmp_fake)


if __name__ == "__main__":
    import torch

    cats = ['apple', 'moon', 'book', 'shark', 'angel', 'bicycle']

    for c_cat in cats:

        # 设置你的目录路径
        real_images_path = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\quichdraw\{c_cat}'

        fake_images_path_sdgraph = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sdgraph\{c_cat}_11_16'
        fake_images_path_sketchknitter = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sketchknitter\{c_cat}'
        fake_images_path_sketchrnn = rf'E:\document\deeplearning_idea\sketch temporal is out\fid_cal\sketchrnn\{c_cat}'

        fid_1 = compute_fid(real_images_path, fake_images_path_sdgraph)
        print(c_cat, f'sdgraph: {fid_1}')

        fid_2 = compute_fid(real_images_path, fake_images_path_sketchknitter)
        print(c_cat, f'sketchknitter: {fid_2}')

        fid_3 = compute_fid(real_images_path, fake_images_path_sketchrnn)
        print(c_cat, f'sketchrnn: {fid_3}')





