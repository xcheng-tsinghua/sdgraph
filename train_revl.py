"""
训练检索
"""
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
import argparse
from data_utils.sketch_dataset import RetrievalDataset
from encoders.sketch_rnn import SketchRNNEmbedding as SketchEncoder
from tqdm import tqdm
from colorama import Fore, Back, init
import numpy as np
from itertools import chain
from datetime import datetime

from encoders.vit import VITFinetune
from encoders.utils import inplace_relu, clear_log, clear_confusion, all_metric_cls, get_log, get_false_instance


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--save_str', type=str, default='sketch_rnn', help='---')

    parser.add_argument('--bs', type=int, default=10, help='batch size in training')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_retrieval\test_dataset')

    '''
    parser.add_argument('--root_sever', type=str, default=f'/root/my_data/data_set/unified_sketch_from_quickdraw/stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',  help='root of dataset')
    parser.add_argument('--root_local', type=str, default=f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='root of dataset')
    '''

    return parser.parse_args()


def constructive_loss(x, y, margin=1.0, lambda_=1.0):
    """
    对比损失
    :param x: [bs ,emb]
    :param y: [bs ,emb]
    :param margin:
    :param lambda_:
    :return:
    """
    # x, y: tensors of shape (N, D)
    N = x.size(0)

    # 计算对应行之间的距离
    pos_dist = F.pairwise_distance(x, y, p=2)
    pos_loss = torch.mean(pos_dist ** 2)

    # 计算 x 与 y 中所有不同行之间的距离
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, N, D)
    dist_matrix = torch.norm(x_exp - y_exp, dim=2, p=2)  # (N, N)

    # 创建掩码，排除对角线（即对应行）
    mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
    neg_dist = dist_matrix[mask]

    # 计算不同行之间的损失
    neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)

    # 总损失
    loss = pos_loss + lambda_ * neg_loss
    return loss


class EmbeddingSpace(object):
    """
    创建一个特征集合，包含测试集中全部数据
    """
    def __init__(self,
                 img_encoder: Module,
                 skh_img_loader: DataLoader
                 ):

        img_encoder = img_encoder.eval()

        self.embeddings = []
        self.data_idx = []
        for data in tqdm(skh_img_loader, total=len(skh_img_loader), desc='embeding'):
            images, idxes = data[0].float().cuda(), data[1].long().cuda()

            self.data_idx.append(idxes)

            with torch.no_grad():
                # -> [bs, emb]
                img_embedding = img_encoder(images)

                # 获取测试集中全部的图片 embedding，这里embedding的索引即对应它的文件
                self.embeddings.append(img_embedding)

        # -> [all, emb]
        self.embeddings = torch.cat(self.embeddings, dim=0)

        # -> [all, ]
        self.data_idx = torch.cat(self.data_idx, dim=0)

    def top_k(self, sketches: torch.Tensor, k: int):
        """
        :param sketches: [bs, emb]
        :param k:
        :return: [bs, k]
        """
        # -> [bs, all]
        distances = torch.cdist(sketches, self.embeddings)

        # -> [bs, k]
        topk_indices = torch.topk(distances, k, largest=False, dim=1)[1]

        searched_idx = self.data_idx[topk_indices]
        return searched_idx


def test(img_encoder, skh_encoder, skh_img_dataset, skh_img_loader):
    skh_img_dataset.img()
    emb_space = EmbeddingSpace(img_encoder, skh_img_loader)

    c_correct_1 = 0
    c_correct_5 = 0
    c_correct_10 = 0
    c_all = 0

    skh_img_dataset.eval()
    skh_encoder = skh_encoder.eval()
    for idx_batch, data in tqdm(enumerate(skh_img_loader), total=len(skh_img_loader), desc='evaluate'):
        sketch, mask, v_index = data[0].float().cuda(), data[1].float().cuda(), data[3].long().cuda()

        with torch.no_grad():
            # [bs, emb]
            skh_embedding = skh_encoder(sketch, mask)

            # [bs, k]
            searched_idx = emb_space.top_k(skh_embedding, 10)

            # 计算准确率
            matches = (searched_idx == v_index.unsqueeze(1))  # [bs, k]

            # 检查每行是否至少有一个 True
            match_1 = matches[:, 0].sum().item()  # [bs] 布尔向量

            match_5 = matches[:, :5].any(dim=1).sum().item()  # [bs] 布尔向量

            match_10 = matches.any(dim=1).sum().item()

            c_correct_1 += match_1
            c_correct_5 += match_5
            c_correct_10 += match_10
            c_all += sketch.size(0)

    return c_correct_1 / c_all, c_correct_5 / c_all, c_correct_10 / c_all


def main(args):
    """
    训练函数
    :param args:
    :return:
    """

    '''日志记录'''
    logger = get_log('./log/' + args.save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')

    # 设置数据集
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    dataset = RetrievalDataset(root=data_root, back_mode='S5')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

    '''加载模型及权重'''
    img_encoder = VITFinetune().cuda()
    skh_encoder = SketchEncoder().cuda()

    skh_weight_path = './model_trained/sketch_retrieval_skh.pth'
    img_weight_path = './model_trained/sketch_retrieval_img.pth'
    if args.is_load_weight == 'True':
        try:
            img_encoder.load_state_dict(torch.load(img_weight_path))
            print(Fore.GREEN + 'training image encoder from exist model: ' + img_weight_path)

            skh_encoder.load_state_dict(torch.load(skh_weight_path))
            print(Fore.GREEN + 'training sketch encoder from exist model: ' + skh_weight_path)

        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        chain(img_encoder.parameters(), list(skh_encoder.parameters())),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    for epoch in range(args.epoch):
        skh_encoder = skh_encoder.train()
        img_encoder = img_encoder.train()
        loss_all = []

        dataset.train()
        for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='training'):
            skh, mask, img = data[0].float().cuda(), data[1].float().cuda(), data[2].float().cuda()

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            skh_emb = skh_encoder(skh, mask)
            img_emb = img_encoder(img)

            loss = constructive_loss(skh_emb, img_emb)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())

        scheduler.step()
        torch.save(skh_encoder.state_dict(), skh_weight_path)
        torch.save(img_encoder.state_dict(), img_weight_path)

        # print(f'save sketch weights at: {skh_weight_path}')
        # print(f'save image weights at: {img_weight_path}')
        # print(f'{epoch} / {args.epoch}: loss: {np.mean(loss_all)}')

        acc_top1, acc_top5, acc_top10 = test(img_encoder, skh_encoder, dataset, dataloader)
        eval_str = f'{args.save_str}:{epoch}/{args.epoch}-loss: {np.mean(loss_all)} acc_top1: {acc_top1} acc_top5: {acc_top5} acc_top10: {acc_top10}'
        print(eval_str)
        logger.info(eval_str)


if __name__ == '__main__':
    clear_log('./log')
    clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    main(parse_args())

