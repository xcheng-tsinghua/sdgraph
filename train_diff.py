# 工具包
import torch
import os
import logging # 记录日志信息
import argparse
from colorama import Fore, Back, init
from datetime import datetime

# 自建模块
import global_defs
from data_utils.SketchDataset import DiffDataset
from data_utils.sketch_vis import save_format_sketch
from encoders.sdgraph import SDGraphSeg as SDGraphSeg
from GaussianDiffusion import GaussianDiffusion
from encoders.utils import clear_log


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--save_str', type=str, default='sdgraph_unet_more', help='---')

    parser.add_argument('--bs', type=int, default=128, help='batch size in training')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--n_skgen', default=30, type=int, help='---')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=f'/root/my_data/data_set/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='root of dataset')
    parser.add_argument('--root_local', type=str, default=f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='root of dataset')

    return parser.parse_args()


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    os.makedirs('model_trained/', exist_ok=True)
    os.makedirs('imgs_gen/', exist_ok=True)
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''定义数据集'''
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever
    train_dataset = DiffDataset(root=data_root, shuffle_stk=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    '''加载模型及权重'''
    model = SDGraphSeg(2, 2)
    model_savepth = 'model_trained/' + save_str + '.pth'

    if args.is_load_weight == 'True':
        try:
            model.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    diffusion = GaussianDiffusion(model, model.pnt_channel(), global_defs.n_skh_pnt)
    diffusion = diffusion.cuda()

    '''优化器'''
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    for epoch_idx in range(args.epoch):
        diffusion = diffusion.train()

        print(f'Epoch ({epoch_idx + 1}/{args.epoch}):')

        for batch_idx, data in enumerate(train_dataloader, 0):
            points = data.float().cuda().permute(0, 2, 1)  # -> [bs, 2, n_points]

            optimizer.zero_grad()
            loss = diffusion(points)
            loss.backward()
            optimizer.step()

            state_str = f"Epoch {epoch_idx + 1}/{args.epoch}:, batch_idx {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.detach().item():.4f}"
            print(state_str)
            logger.info(state_str)

        scheduler.step()
        torch.save(model.state_dict(), 'model_trained/' + save_str + '.pth')

    '''生成图片'''
    with torch.no_grad():
        diffusion = diffusion.eval()

        print('generate images')
        sample_epoch = args.n_skgen // 10
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


