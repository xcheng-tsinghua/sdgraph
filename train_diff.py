import os
import sys

# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# 将models文件夹的路径添加到sys.path中，使得models文件夹中的py文件能被本文件import
sys.path.append(os.path.join(ROOT_DIR, 'encoders'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

# 工具包
import torch
from datetime import datetime
import logging # 记录日志信息
import argparse
from colorama import Fore, Back, init

# 自建模块
from data_utils.SketchDataset import DiffDataset
from data_utils.sketch_vis import save_format_sketch
from bks.SDGraph import SDGraphUNet
from GaussianDiffusion import GaussianDiffusion


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--epoch', default=000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--is_load_weight', type=str, default='True', choices=['True', 'False'], help='---')
    parser.add_argument('--n_figgen', default=30, type=int, help='---')
    parser.add_argument('--local', default='True', choices=['True', 'False'], type=str, help='---')

    parser.add_argument('--save_str', type=str, default='sdgraph_unet_valid', help='---')
    parser.add_argument('--root_sever', type=str, default=r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32_no_mix_proc', help='root of dataset')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk4_stkpnt32_no_mix_proc', help='root of dataset')

    # 参数化数据集：D:/document/DeepLearning/DataSet/data_set_p2500_n10000
    # 机械草图数据集（服务器）：r'/root/my_data/data_set/unified_sketch'
    # 机械草图数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch_simplify2'
    # 机械草图数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch'
    # quickdraw 数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk5_stkpnt32'
    # quickdraw 数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk10_stkpnt64'
    # quickdraw 数据集（服务器）：r'/root/my_data/data_set/unified_sketch_from_quickdraw/apple_stk5_stkpnt32'
    return parser.parse_args()


def clear_log(folder_path, k=5):
    """
    遍历文件夹内的所有 .txt 文件，删除行数小于 k 的文件。

    :param folder_path: 要处理的文件夹路径
    :param k: 行数阈值，小于 k 的文件会被删除
    """
    os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查是否为 .txt 文件
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # 统计文件的行数
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    num_lines = len(lines)

                # 如果行数小于 k，则删除文件
                if num_lines < k:
                    print(f"Deleting file: {file_path} (contains {num_lines} lines)")
                    os.remove(file_path)
            except Exception as e:
                # 捕获读取文件时的错误（如编码问题等）
                print(f"Error reading file {file_path}: {e}")


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 定义数据集，训练集及对应加载器
    train_dataset = DiffDataset(root=args.root)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    '''MODEL LOADING'''
    model = SDGraphUNet()

    os.makedirs('model_trained/', exist_ok=True)
    model_savepth = 'model_trained/' + save_str + '.pth'

    if args.is_load_weight == 'True':
        try:
            model.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    diffusion = GaussianDiffusion(model)
    diffusion = diffusion.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate, # 0.001
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate # 1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''TRANING'''
    for epoch_idx in range(args.epoch):
        diffusion = diffusion.train()

        print(f'Epoch ({epoch_idx + 1}/{args.epoch}):')

        for batch_idx, data in enumerate(trainDataLoader, 0):

            # -> [bs, n_points, 2]
            points = data.float().cuda()

            # -> [bs, 2, n_points]
            points = points.permute(0, 2, 1)

            channel = points.size()[1]
            assert channel == 2

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            loss = diffusion(points)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            state_str = f"Epoch {epoch_idx + 1}/{args.epoch}:, batch_idx {batch_idx + 1}/{len(trainDataLoader)}, Loss: {loss.detach().item():.4f}"
            print(state_str)
            logger.info(state_str)

        scheduler.step()
        torch.save(model.state_dict(), 'model_trained/' + save_str + '.pth')

    # 推理部分
    with torch.no_grad():
        diffusion = diffusion.eval()

        print('generate images')
        sample_epoch = args.n_figgen // 10
        gen_idx = 0
        for i in range(sample_epoch):
            print(f'generate {i * 10} to {(i + 1) * 10 - 1}')

            sampled_images = diffusion.sample(batch_size=10)
            for batch_fig_idx in range(10):
                save_format_sketch(sampled_images[batch_fig_idx, :, :], f'imgs_gen/{save_str}-{gen_idx}.png')
                gen_idx += 1


if __name__ == '__main__':
    clear_log('./log')
    init(autoreset=True)
    main(parse_args())


