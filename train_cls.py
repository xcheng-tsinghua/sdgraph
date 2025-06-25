# 工具包
import torch
import torch.nn.functional as F
from datetime import datetime
import argparse
from tqdm import tqdm
from colorama import Fore, Back, init
import os
import time
import multiprocessing as mp

# 自建模块
from data_utils.sketch_dataset import QuickDrawCls, SketchDatasetCls
# from encoders.sdgraph_stk_samp import SDGraphCls
# from encoders.sdgraph import SDGraphCls
from encoders.sdgraph_ablation_sg import SDGraphCls
from encoders.sketch_transformer import SketchTransformerCls
from encoders.sketch_rnn import SketchRNN_Cls
from encoders.utils import inplace_relu, clear_log, clear_confusion, all_metric_cls, get_log, get_false_instance
import global_defs


def parse_args():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=100, help='batch size in training')
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--is_load_weight', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--is_shuffle_stroke', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--model', type=str, default='SketchRNN', choices=['SketchRNN', 'SketchTransformer', 'SDGraph'])

    parser.add_argument('--save_str', type=str, default=f'sdgraph_{global_defs.n_stk}_{global_defs.n_stk_pnt}')
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/quickdraw/MGT/random')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean')

    r'''
    cad sketch
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_cad/sketch_txt_all')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all')
    
    TuBerlin_raw
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/TU_Berlin/raw/svg')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_raw\svg')
    
    QuickDraw:
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/quickdraw/raw')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\quickdraw\raw')
    
    QuickDraw MGT STK:
    该数据集目录用于 sketchrnn 和 sketchtransformer 的训练
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/quickdraw/MGT/log_normal_mean')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\quickdraw\MGT\log_normal_mean')
    
    QuickDraw MGT STK:
    该数据集目录用于 sdgraph 的训练
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/quickdraw/mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\quickdraw\mgt_normal_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}')
    
    '''
    return parser.parse_args()


def main(args):
    if args.model == 'SDGraph':
        save_str = f'{args.model.lower()}_SG_{global_defs.n_stk}_{global_defs.n_stk_pnt}'
    else:
        save_str = args.model.lower()

    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    model_savepth = 'model_trained/best_' + save_str + '.pth'
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')

    '''定义数据集'''
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    if args.model == 'SDGraph':
        back_mode = 'STK'
    else:
        back_mode = 'S5'

    if args.is_shuffle_stroke == 'True':
        is_shuffle_stroke = True
    else:
        is_shuffle_stroke = False

    dataset = SketchDatasetCls(data_root, back_mode=back_mode, is_already_divided=True, is_preprocess=False, is_shuffle_stroke=is_shuffle_stroke)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)

    '''加载模型及权重'''
    if args.model == 'SketchRNN':
        classifier = SketchRNN_Cls(dataset.n_classes()).cuda()

    elif args.model == 'SketchTransformer':
        classifier = SketchTransformerCls(dataset.n_classes()).cuda()

    elif args.model == 'SDGraph':
        classifier = SDGraphCls(dataset.n_classes(), 2).cuda()

    else:
        raise TypeError('error model type')

    if args.is_load_weight == 'True':
        try:
            classifier.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    '''训练'''
    best_instance_accu = -1.0
    time_all = []
    for epoch in range(args.epoch):
        logstr_epoch = f'Epoch({epoch}/{args.epoch}):'
        all_preds = []
        all_labels = []

        classifier = classifier.train()
        dataset.train()
        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            points, mask, target = data[0].float().cuda(), data[1].float().cuda(), data[2].long().cuda()

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            # 模型传入数据，获取输出，并计算loss
            pred = classifier(points, mask)
            loss = F.nll_loss(pred, target)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            # 保存数据用于计算指标
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        # 计算分类指标
        all_metric_train = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'train-{epoch}.png'))
        logstr_trainaccu = f'\ttrain_instance_accu:\t{all_metric_train[0]}'

        # 调整学习率并保存权重
        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        '''测试'''
        with torch.no_grad():
            all_preds = []
            all_labels = []
            # all_indexes = []

            classifier = classifier.eval()
            dataset.eval()
            start_time = time.time()
            for j, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                points, mask, target = data[0].float().cuda(), data[1].float().cuda(), data[2].long().cuda()

                pred = classifier(points, mask)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

                # 保存索引用于计算分类错误的实例
                # all_indexes.append(data[-1].long().detach().cpu().numpy())
            end_time = time.time()
            avg_time = (end_time - start_time) / len(dataloader)
            time_all.append(avg_time)

            all_metric_eval = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'eval-{epoch}.png'))
            accustr = f'\teval_ins_acc\t{all_metric_eval[0]}\teval_cls_acc\t{all_metric_eval[1]}\teval_f1_m\t{all_metric_eval[2]}\teval_f1_w\t{all_metric_eval[3]}\tmAP\t{all_metric_eval[4]}'
            logger.info(logstr_epoch + logstr_trainaccu + accustr)

            # get_false_instance(all_preds, all_labels, all_indexes, test_dataset)

            print(f'{save_str}: epoch {epoch}/{args.epoch}: train_ins_acc: {all_metric_train[0]}, test_ins_acc: {all_metric_eval[0]}, cinfer_time: {avg_time}, all_infer_time: {sum(time_all) / len(time_all)}')

            # 额外保存最好的模型
            if best_instance_accu < all_metric_eval[0]:
                best_instance_accu = all_metric_eval[0]
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    clear_log('./log')
    clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    mp.set_start_method('spawn', force=True)
    main(parse_args())

