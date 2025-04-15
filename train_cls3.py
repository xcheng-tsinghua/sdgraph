# 工具包
import torch
import torch.nn.functional as F
from datetime import datetime
import argparse
from tqdm import tqdm
from colorama import Fore, Back, init
import os
import numpy as np

# 自建模块
import global_defs
from data_utils.SketchDataset import SketchDataset2 as SketchDataset
from encoders.sdgraph3 import SDGraphClsTest as SDGraphCls
# from encoders.sdgraph import SDGraphCls
from encoders.utils import inplace_relu, clear_log, clear_confusion, all_metric_cls, get_log


def has_nan_weight(module):
    for name, param in module.named_parameters():
        if param is not None and torch.isnan(param).any():
            # print(f"参数{name}中存在NaN")
            return True
    # print("该模块所有参数均无NaN")
    return False


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=90, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')

    parser.add_argument('--save_str', type=str, default='sdgraph_test', help='---')

    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_cad/sketch_txt', help='---')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt', help='---')

    r'''
    cad sketch
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_cad/sketch_txt', help='---')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt', help='---')
    
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_cad/unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='---')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_cad\unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='---')
    TuBerlin
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/TU_Berlin/TU_Berlin_txt_cls', help='---')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\TU_Berlin\TU_Berlin_txt_cls', help='---')
    '''

    return parser.parse_args()


def get_false_instance(all_preds: list, all_labels: list, all_indexes: list, dataset, save_path: str = './log/false_instance.txt'):
    """
    获取全部分类错误的实例路径
    :param all_preds:
    :param all_labels:
    :param all_indexes:
    :param dataset:
    :param save_path:
    :return:
    """
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [n_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [n_samples]
    all_indexes = np.hstack(all_indexes)  # 形状为 [n_samples]

    # 确保all_labels, all_indexes中保存的为整形数据
    assert np.issubdtype(all_labels.dtype, np.integer) and np.issubdtype(all_indexes.dtype, np.integer)

    all_preds = np.argmax(all_preds, axis=1)  # -> [n_samples, ]
    incorrect_index = np.where(all_preds != all_labels)[0]
    incorrect_index = all_indexes[incorrect_index]
    incorrect_preds = all_preds[incorrect_index]

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            for c_idx, c_data_idx in enumerate(incorrect_index):
                # 找到分类错误的类型：
                false_class = ''
                for k, v in dataset.classes.items():
                    if incorrect_preds[c_idx] == v:
                        false_class = k
                        break

                f.write(dataset.datapath[c_data_idx][1] + ' | ' + false_class + '\n')

        print('save incorrect cls instance: ', save_path)


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    model_savepth = 'model_trained/' + save_str + '.pth'
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')

    '''定义数据集'''
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = SketchDataset(root=data_root, is_train=True)
    test_dataset = SketchDataset(root=data_root, is_train=False)
    num_class = len(train_dataset.classes)

    # sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=64, replacement=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    # sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=64, replacement=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    '''加载模型及权重'''
    classifier = SDGraphCls(num_class).cuda()
    # classifier = PointNet2(num_class)
    # classifier = DGCNN(num_class)
    # classifier = Attention(num_class)
    # classifier = pointnet(num_class)
    # loss_func = get_loss().cuda()

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
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    best_instance_accu = -1.0
    for epoch in range(args.epoch):

        '''TRAINING'''
        classifier = classifier.train()

        logstr_epoch = f'Epoch({epoch}/{args.epoch}):'
        all_preds = []
        all_labels = []

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            # points, mask, target = data[0].float().cuda(), data[1].bool().cuda(), data[2].long().cuda()
            points, target = data[0].float().cuda(), data[1].long().cuda()
            # stk_coor = data[2].float().cuda()  # [bs, n_stk, 512]
            # assert stk_coor.size(1) == global_defs.n_stk

            # -> [bs, 2, n_points]
            # points = points.permute(0, 2, 1)
            # assert points.size()[1] == 2

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            # pred = classifier(points, mask)
            pred = classifier(points)
            loss = F.nll_loss(pred, target)

            assert not has_nan_weight(classifier)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            assert not has_nan_weight(classifier)

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
            classifier = classifier.eval()

            all_preds = []
            all_labels = []
            # all_indexes = []

            for j, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                # points, mask, target = data[0].float().cuda(), data[1].bool().cuda(), data[2].long().cuda()
                points, target = data[0].float().cuda(), data[1].long().cuda()
                # stk_coor = data[2].float().cuda()  # [bs, n_stk, 512]
                # assert stk_coor.size(1) == global_defs.n_stk

                # points = points.permute(0, 2, 1)
                # assert points.size()[1] == 2

                # pred = classifier(points, mask)
                pred = classifier(points)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

                # 保存索引用于计算分类错误的实例
                # all_indexes.append(data[-1].long().detach().cpu().numpy())

            all_metric_eval = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'eval-{epoch}.png'))
            accustr = f'\teval_ins_acc\t{all_metric_eval[0]}\teval_cls_acc\t{all_metric_eval[1]}\teval_f1_m\t{all_metric_eval[2]}\teval_f1_w\t{all_metric_eval[3]}\tmAP\t{all_metric_eval[4]}'

            # 额外保存最好的模型
            if best_instance_accu < all_metric_eval[0]:
                best_instance_accu = all_metric_eval[0]
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')

        # save log
        logger.info(logstr_epoch + logstr_trainaccu + accustr)
        print(f'{save_str}: epoch {epoch}/{args.epoch}: train_ins_acc: {all_metric_train[0]}, test_ins_acc: {all_metric_eval[0]}')
        # get_false_instance(all_preds, all_labels, all_indexes, test_dataset)


if __name__ == '__main__':
    clear_log('./log')
    clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    main(parse_args())


