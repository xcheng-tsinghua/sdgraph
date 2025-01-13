
# 工具包
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from colorama import Fore, Back, Style, init
import shutil
import os

# 自建模块
from data_utils.SketchDataset import SketchDataset
from SDGraphCls import SDGraph
from data_utils.sketch_utils import save_confusion_mat
from encoders.PointNet2 import PointNet2
from encoders.Dgcnn import DGCNN
from encoders.Attention import Attention
from encoders.pointnet_cls import pointnet, get_loss


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--batch_size', '-B', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', '-E', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', '-R', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', '-D', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--is_load_weight', '-W', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--local', '-L', default='True', choices=['True', 'False'], type=str, help='---')

    parser.add_argument('--save_str', '-S', type=str, default='sdgraph', help='---')
    parser.add_argument('--root_sever', '-RS', type=str, default=r'/root/my_data/data_set/unified_sketch', help='---')
    parser.add_argument('--root_local', '-RL', type=str, default=r'D:\document\DeepLearning\DataSet\unified_sketch', help='---')

    # 参数化数据集：D:/document/DeepLearning/DataSet/data_set_p2500_n10000
    # 机械草图数据集（服务器）：r'/root/my_data/data_set/unified_sketch'
    # 机械草图数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch_simplify2'
    # 机械草图数据集（本地）：r'D:\document\DeepLearning\DataSet\unified_sketch'
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def accuracy_over_class(all_labels: list, all_preds: list, n_classes: int):
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    accuracies = []

    for class_idx in range(n_classes):
        # 找到当前类别的所有样本
        class_mask = (all_labels == class_idx)
        # 如果当前类别的样本数为0，则跳过
        if class_mask.sum().item() == 0:
            continue
        # 计算当前类别的准确率
        class_accuracy = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
        accuracies.append(class_accuracy)

    # 返回所有类别准确率的平均值
    return np.mean(accuracies)


def mean_average_precision(all_labels: list, all_preds: list, num_class: int):
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [total_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [total_samples]

    # 将真实标签转化为one-hot编码 (one-vs-rest)
    all_labels_bin = label_binarize(all_labels, classes=np.arange(num_class))

    # 计算每个类别的AP
    ap_scores = []
    for i in range(num_class):
        ap = average_precision_score(all_labels_bin[:, i], all_preds[:, i])
        ap_scores.append(ap)

    # 计算mAP (所有类别的AP的平均值)
    mAP = np.mean(ap_scores)
    return mAP


def clear_log(log_dir):
    """
    清空空白的log文件
    """
    os.makedirs(log_dir, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(log_dir):
        # 获取文件的完整路径
        file_path = os.path.join(log_dir, filename)
        # 检查是否为txt文件且为空
        if filename.endswith('.txt') and os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")


def clear_confusion(root_dir='./data_utils/confusion', k=5):
    """
    遍历 root_dir 中的文件夹，删除文件数小于 k 的文件夹。

    :param root_dir: 根目录
    :param k: 文件数的阈值，小于 k 的文件夹会被删除
    """
    for foldername, subfolders, filenames in os.walk(root_dir, topdown=False):
        # 遍历每个文件夹
        num_files = len(filenames)
        if num_files < k:
            # 如果文件数小于 k，则删除整个文件夹
            print(f"Deleting folder: {foldername} (contains {num_files} files)")
            shutil.rmtree(foldername)


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)

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

    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    # 定义数据集，训练集及对应加载器
    train_dataset = SketchDataset(root=data_root, is_train=True)
    test_dataset = SketchDataset(root=data_root, is_train=False)
    num_class = len(train_dataset.classes)

    # sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=64, replacement=False)
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    # sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=64, replacement=False)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 获取分类模型
    classifier = SDGraph(num_class)
    # classifier = PointNet2(num_class)
    # classifier = DGCNN(num_class)
    # classifier = Attention(num_class)
    # classifier = pointnet(num_class)
    # loss_func = get_loss().cuda()

    os.makedirs('model_trained/', exist_ok=True)
    model_savepth = 'model_trained/' + save_str + '.pth'

    if args.is_load_weight == 'True':
        try:
            classifier.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    classifier = classifier.cuda()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate, # 0.001
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate # 1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_instance_accu = -1.0

    '''TRANING'''
    for epoch in range(args.epoch):
        classifier = classifier.train()

        logstr_epoch = f'Epoch ({epoch + 1}/{args.epoch}):'

        mean_correct = []
        pred_cls = []
        target_cls = []

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):

            # -> [bs, n_points, 2]
            points = data[0].float().cuda()

            # -> [bs, 2, n_points]
            points = points.permute(0, 2, 1)

            channel = points.size()[1]
            assert channel == 2

            # -> [bs, ]
            target = data[1].long().cuda()

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            pred = classifier(points)
            loss = F.nll_loss(pred, target)
            # loss = loss_func(pred[0], target, pred[1])
            # pred = pred[0]

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            pred_cls += pred_choice.tolist()
            target_cls += target.tolist()

        save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'train-{epoch}.png'))

        acc_over_class = accuracy_over_class(target_cls, pred_cls, num_class)
        logstr_trainaccu = f'\ttrain_instance_accu:\t{np.mean(mean_correct)}\ttrain_class_accu:\t{acc_over_class}'

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            classifier = classifier.eval()

            # 计算Acc.OverInstance
            total_correct = 0
            total_testset = 0

            # 保存ConfusionMatrix和计算Acc.OverClass
            pred_cls = []
            target_cls = []

            # 计算mAP
            all_preds = []
            all_labels = []

            for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points = data[0].float().cuda()
                target = data[1].long().cuda()

                points = points[:, :, :2]
                points = points.permute(0, 2, 1)
                assert points.size()[1] == 2

                pred = classifier(points)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

                total_correct += correct.item()
                total_testset += points.size()[0]

                pred_cls += pred_choice.tolist()
                target_cls += target.tolist()

            save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'eval-{epoch}.png'))

            # 计算 mean average precision
            mAP = mean_average_precision(all_labels, all_preds, num_class)

            # 计算 Acc. over Instance
            acc_over_instance = total_correct / float(total_testset)

            # 计算 Acc. over Class
            acc_over_class = accuracy_over_class(target_cls, pred_cls, num_class)

            # 计算 F1-Score
            macro_f1_score = f1_score(target_cls, pred_cls, average='macro')
            weighted_f1_score = f1_score(target_cls, pred_cls, average='weighted')

            accustr = f'\ttest_instance_accuracy\t{acc_over_instance}\ttest_class_accuracy\t{acc_over_class}\ttest_F1_Score\t{macro_f1_score}\tmAP\t{mAP}\twmAP\t{weighted_f1_score}'
            logger.info(logstr_epoch + logstr_trainaccu + accustr)
            print(accustr.replace('\t', ' '))

            # 额外保存最好的模型
            if best_instance_accu < acc_over_class:
                best_instance_accu = acc_over_class
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    clear_log('./log')
    clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    main(parse_args())


