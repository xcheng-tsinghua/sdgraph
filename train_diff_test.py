# 工具包
import torch
import os
import logging # 记录日志信息
import argparse
from colorama import Fore, Back, init
from datetime import datetime

# 自建模块
import global_defs
from data_utils.sketch_dataset import DiffDataset
from data_utils.vis import save_format_sketch_test
from encoders.sdgraph_test import SDGraphUNet
from GaussianDiffusion_test import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--save_str', type=str, default='sdgraph_ext_interp_mse_nll', help='---')

    parser.add_argument('--bs', type=int, default=40, help='batch size in training')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--n_skh_gen', default=300, type=int, help='---')
    parser.add_argument('--n_print_skip', default=10, type=int, help='print batch loss after n_print_skip batch number')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=fr'/opt/data/private/data_set/quickdraw/stk2/book_stk_16_32_ext_interp', help='root of dataset')
    parser.add_argument('--root_local', type=str, default=fr'D:\document\DeepLearning\DataSet\quickdraw\stk2\book_stk_16_32_ext_interp', help='root of dataset')

    return parser.parse_args()


def main(args):
    print(args)

    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    skh_save_folder = os.path.join('imgs_gen', f'{args.save_str}_{global_defs.n_stk}_{global_defs.n_stk_pnt}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(skh_save_folder, exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''加载模型及权重'''
    model = SDGraphUNet()
    model_savepth = 'model_trained/' + save_str + '.pth'

    if args.is_load_weight == 'True':
        try:
            model.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.RED + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    diffusion = GaussianDiffusion(model, model.img_size())
    diffusion = diffusion.cuda()

    if args.epoch > 0:

        '''定义数据集'''
        if args.local == 'True':
            data_root = args.root_local
        else:
            data_root = args.root_sever
        train_dataset = DiffDataset(root=data_root, is_stk_processed=True, delimiter=' ')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

        '''优化器'''
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        '''训练'''
        for epoch_idx in range(args.epoch):
            diffusion = diffusion.train()

            print(f'Epoch ({epoch_idx + 1}/{args.epoch}):')

            for batch_idx, data in enumerate(train_dataloader, 0):
                points = data[0].float().cuda()  # -> [bs, n_points, 3]

                col3 = points[..., 2].unsqueeze(-1)
                inv = 1 - points[..., 2].unsqueeze(-1)
                points = torch.cat([points[..., :2], inv, col3], dim=-1)

                optimizer.zero_grad()
                loss, mse, nll = diffusion(points)
                loss.backward()
                optimizer.step()

                state_str = f"Epoch {epoch_idx}/{args.epoch}:, batch_idx {batch_idx}/{len(train_dataloader)}, mse: {mse:.4f}, nll: {nll:.4f}"
                if batch_idx % args.n_print_skip == 0:
                    print(state_str)
                logger.info(state_str)

            scheduler.step()
            torch.save(model.state_dict(), 'model_trained/' + save_str + '.pth')

    '''生成图片'''
    with torch.no_grad():
        print('sketch save to: ', os.path.abspath(skh_save_folder))
        diffusion = diffusion.eval()

        print('generate images')
        sample_epoch = args.n_skh_gen // 10
        gen_idx = 0
        for i in range(sample_epoch):
            print(f'generate {i * 10} to {(i + 1) * 10 - 1}')

            sampled_images = diffusion.sample(batch_size=10)
            for batch_fig_idx in range(10):
                skh_save_name = os.path.join(skh_save_folder, f'{save_str}-{gen_idx}.png')
                save_format_sketch_test(sampled_images[batch_fig_idx], skh_save_name)
                # save_format_sketch_ext(sampled_images[batch_fig_idx], skh_save_name,
                #                    is_near_merge=True, retreat=(0, 1), merge_dist=args.scale * 0.10)
                gen_idx += 1


if __name__ == '__main__':
    # clear_log('./log')
    init(autoreset=True)
    main(parse_args())


