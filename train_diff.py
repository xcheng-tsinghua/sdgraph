"""
训练 diffusion，一般来说 category = bicycle, n_stk=11, n_stk_pnt=16
"""
# 工具包
import torch
import os
import argparse
from colorama import Fore, Back, init
from datetime import datetime
import shutil

# 自建模块
import global_defs
from data_utils.sketch_dataset import DiffDataset, QuickDrawDiff
from data_utils.vis import save_format_sketch
# from encoders.sdgraph_stk_samp import SDGraphUNet as sd_stk_sample
from encoders.sdgraph_stk_samp_endsnap import SDGraphUNet as sd_stk_sample
# from encoders.sdgraph_test import SDGraphUNet as sd_stk_sample
# from encoders.sdgraph import SDGraphUNet as sd_normal
from encoders.sdgraph_endsnap import SDGraphUNet as sd_normal
# from ablation.sdgraph_endsnap_nomix import SDGraphUNet as sd_normal
from GaussianDiffusion import GaussianDiffusion
from encoders.utils import clear_log, get_log


def parse_args():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=40, help='batch size in training')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='True', choices=['True', 'False'], help='---')
    parser.add_argument('--n_skh_gen', default=30, type=int, help='---')
    parser.add_argument('--n_print_skip', default=10, type=int, help='print batch loss after n_print_skip batch number')
    parser.add_argument('--scale', default=100, type=float, help='sketch bonding box is within [-scale, scale]')

    parser.add_argument('--category', default='bicycle', type=str, help='training diffusion category')
    parser.add_argument('--is_stk_sample', default='False', type=str, choices=['True', 'False'], help='using stroke sample model?')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='running on local?')
    parser.add_argument('--is_load_npz', default='False', type=str, choices=['True', 'False'], help='using quickdraw npz file?')
    parser.add_argument('--root_sever', type=str, default=fr'/opt/data/private/data_set/quickdraw/stk2/book_stk_16_32_auto_space_snap')
    parser.add_argument('--root_local', type=str, default=fr'D:\document\DeepLearning\DataSet\quickdraw\stk2\book_stk_16_32_auto_space_snap')

    r'''
    parser.add_argument('--root_sever', type=str, default=f'/root/my_data/data_set/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',  help='root of dataset')
    parser.add_argument('--root_local', type=str, default=f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='root of dataset')
    
    parser.add_argument('--root_sever', type=str, default=fr'/root/my_data/data_set/quickdraw/raw/apple.full.npz')
    parser.add_argument('--root_local', type=str, default=fr'D:\document\DeepLearning\DataSet\quickdraw\raw\apple.full.npz')
    
    已预处理好数据
    parser.add_argument('--root_sever', type=str, default=fr'/root/my_data/data_set/quickdraw/diffusion/apple_{global_defs.n_stk}_{global_defs.n_stk_pnt}')
    parser.add_argument('--root_local', type=str, default=fr'D:\document\DeepLearning\DataSet\quickdraw\diffusion\apple_{global_defs.n_stk}_{global_defs.n_stk_pnt}')
    
    '''
    return parser.parse_args()


def main(args):
    print(args)

    save_str = f'sdgraph_{args.category}_{global_defs.n_stk}_{global_defs.n_stk_pnt}'
    # save_str = args.save_str.replace('$TYPE$', args.category)

    if args.is_stk_sample == 'True':
        model = sd_stk_sample(2, 2)
        save_str = save_str.replace('sdgraph', 'sdgraph_stk_sample')
    else:
        model = sd_normal(2, 2)

    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    skh_save_folder = os.path.join('imgs_gen', f'{args.category}_{global_defs.n_stk}_{global_defs.n_stk_pnt}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    print('sketch save to: ', os.path.abspath(skh_save_folder))
    os.makedirs(skh_save_folder, exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

    '''加载模型及权重'''
    # model = SDGraphUNet(2, 2)
    model_savepth = 'model_trained/' + save_str + '.pth'

    if args.is_load_weight == 'True':
        try:
            model.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.RED + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    diffusion = GaussianDiffusion(model, model.img_size(), global_defs.n_skh_pnt)
    diffusion = diffusion.cuda()

    if args.epoch > 0:

        '''定义数据集'''
        if args.local == 'True':
            data_root = args.root_local
        else:
            data_root = args.root_sever

        if args.is_load_npz == 'True':
            data_root = os.path.join(data_root, f'{args.category}.full.npz')
            train_dataset = QuickDrawDiff(root=data_root, workers=0)

        else:
            # data_root = os.path.join(data_root, f'{args.category}_order_stk_{global_defs.n_stk}_{global_defs.n_stk_pnt}')
            train_dataset = DiffDataset(root=data_root, is_stk_processed=True, scale=args.scale, delimiter=' ')

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

        '''优化器'''
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

        '''训练'''
        for epoch_idx in range(args.epoch):
            diffusion = diffusion.train()

            print(f'Epoch ({epoch_idx}/{args.epoch}):')

            for batch_idx, data in enumerate(train_dataloader, 0):
                points, masks = data[0].float().cuda(), data[1].float().cuda()

                optimizer.zero_grad()
                loss = diffusion(points)
                loss.backward()
                optimizer.step()

                state_str = f"{args.category} Epoch {epoch_idx}/{args.epoch}: batch_idx {batch_idx}/{len(train_dataloader)}, Loss: {loss.detach().item():.4f}"
                if batch_idx % args.n_print_skip == 0:
                    print(state_str)
                logger.info(state_str)

            scheduler.step()
            torch.save(model.state_dict(), 'model_trained/' + save_str + '.pth')

    '''生成图片'''
    with torch.no_grad():
        diffusion = diffusion.eval()

        print('generate images')
        sample_epoch = args.n_skh_gen // 10
        gen_idx = 0
        for i in range(sample_epoch):
            print(f'generate {i * 10} to {(i + 1) * 10 - 1}')

            sampled_images = diffusion.sample(batch_size=10)
            for batch_fig_idx in range(10):
                # save_format_sketch(sampled_images[batch_fig_idx], f'imgs_gen/{save_str}-{gen_idx}.png')
                skh_save_name = os.path.join(skh_save_folder, f'{save_str}-{gen_idx}.png')
                save_format_sketch(sampled_images[batch_fig_idx], skh_save_name,
                                   is_near_merge=True, retreat=(1, 0), merge_dist=args.scale * 0.10)

                gen_idx += 1


if __name__ == '__main__':
    # clear_log('./log')
    init(autoreset=True)
    main(parse_args())


