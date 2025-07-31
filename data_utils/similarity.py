import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.transforms as transforms
import torch.nn.functional as F

from data_utils import data_convert as dc
from data_utils import sketch_utils as du
from encoders_3rd.vit import create_pretrained_VIT


def similarity(img1, img2):
    # 显示可视化（可选）
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); plt.title("Left")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title("Right")
    plt.show()

    # 二值化：将草图图像变为黑白
    def binarize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    bin1 = binarize(img1)
    bin2 = binarize(img2)

    # 获取主要轮廓
    def get_largest_contour(binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    cnt1 = get_largest_contour(bin1)
    cnt2 = get_largest_contour(bin2)

    # 使用 Hu 矩比较
    def get_hu(binary):
        moments = cv2.moments(binary)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu_log

    hu1 = get_hu(bin1)
    hu2 = get_hu(bin2)
    hu_distance = np.linalg.norm(hu1 - hu2)

    # 使用轮廓形状匹配
    shape_match_score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)

    print(f"Hu Moment Distance: {hu_distance:.4f}")
    print(f"Shape Match Score: {shape_match_score:.4f}")


def tensor_to_bin_image(tensor_img):
    """将 [1, H, W] tensor 转为二值化 OpenCV 图像"""
    img_np = tensor_img.squeeze().numpy()
    img_np = (img_np * 255).astype(np.uint8)  # 转为 [0,255]
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_main_contour(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def get_hu_features(bin_img):
    moments = cv2.moments(bin_img)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def compute_similarity(tensor1, tensor2):
    bin1 = tensor_to_bin_image(tensor1)
    bin2 = tensor_to_bin_image(tensor2)

    # 轮廓提取
    cnt1 = get_main_contour(bin1)
    cnt2 = get_main_contour(bin2)
    if cnt1 is None or cnt2 is None:
        return None, None, 0.0

    # Hu矩距离
    hu1 = get_hu_features(bin1)
    hu2 = get_hu_features(bin2)
    hu_dist = np.linalg.norm(hu1 - hu2)

    # 形状匹配分数（越小越好）
    match_score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)

    # 相似度分数（越接近1越相似）
    sim_score = np.exp(-10 * match_score)

    return hu_dist, match_score, sim_score


def show_two_tensors(tensor1, tensor2, titles=('Tensor 1', 'Tensor 2')):
    """
    显示两个大小为(1, H, W)的单通道tensor，作为两个子图
    """
    # 确保输入是CPU上的numpy数组
    img1 = tensor1.squeeze().cpu().numpy()  # (H, W)
    img2 = tensor2.squeeze().cpu().numpy()  # (H, W)

    plt.figure(figsize=(8, 4))

    # 第一个子图
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(titles[0])
    plt.axis('off')

    # 第二个子图
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(titles[1])
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def sketch_similarity_of_vector_and_image(vec_path, img_path):
    vec_tensor = dc.s3_to_tensor_img(vec_path, (224, 224), 2)

    img_tensor = Image.open(img_path).convert('L')  # 转为灰度图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # 自动将 (H, W) 转为 (1, H, W) 并归一化到 [0, 1]
    ])
    img_tensor = transform(img_tensor)

    anet = create_pretrained_VIT(root_ckpt=r'E:\document\DeepLearning\sdgraph\model_trained\weight_image_encoder.pth')

    vec_tensor.unsqueeze_(0)
    vec_tensor.unsqueeze_(0)
    vec_tensor = vec_tensor.repeat(3, 3, 1, 1)

    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.repeat(3, 3, 1, 1)

    afea = anet(vec_tensor)
    bfea = anet(img_tensor)

    # show_two_tensors(vec_tensor, img_tensor)
    print(F.mse_loss(bfea, afea))

    pass


def traverse_dir(vec_dir=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3', img_dir=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_png'):
    all_vec = du.get_allfiles(vec_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # 自动将 (H, W) 转为 (1, H, W) 并归一化到 [0, 1]
    ])

    random.shuffle(all_vec)
    for c_vec in all_vec:
        c_img = c_vec.replace(vec_dir, img_dir)
        c_img = os.path.splitext(c_img)[0] + '.png'

        vec_tensor = dc.s3_to_tensor_img(c_vec, (224, 224), 1)
        img_tensor = Image.open(c_img).convert('L')  # 转为灰度图像
        img_tensor = transform(img_tensor)

        show_two_tensors(vec_tensor, img_tensor)


if __name__ == '__main__':
    # avec = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3\airplane\n02691156_394-5.txt'
    # # aimg = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_png\airplane\n02691156_394-5.png'
    #
    # aimg = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_png\bee\n02206856_1216-8.png'
    # sketch_similarity_of_vector_and_image(avec, aimg)

    traverse_dir()


