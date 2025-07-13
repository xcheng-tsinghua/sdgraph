"""
训练检索
"""
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
import argparse
from data_utils.sketch_dataset import RetrievalDataset
from encoders_3rd.sketch_rnn import SketchRNNEmbedding
from encoders_3rd.sketch_transformer import SketchTransformer
from tqdm import tqdm
from colorama import Fore, Back, init
import numpy as np
from itertools import chain
from datetime import datetime
import matplotlib.pyplot as plt
from data_utils.vis import vis_tensor_map, vis_sketch

from encoders_3rd.vit import VITFinetune, create_pretrained_VIT
from encoders.utils import inplace_relu, clear_log, clear_confusion, all_metric_cls, get_log, get_false_instance


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--save_str', type=str, default='sketch_transformer', help='---')

    parser.add_argument('--bs', type=int, default=3, help='batch size in training')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--max_len', type=int, default=40, help='max sketch sequence length')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy_test')

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


def nt_xent_loss(sketch_feat, image_feat, temperature=0.07):
    """
    计算 NT-Xent 对比损失（双向：sketch->image 和 image->sketch）

    参数：
        sketch_feat: Tensor [bs, emb] - 草图特征
        image_feat:  Tensor [bs, emb] - 图像特征
        temperature: 温度参数（通常为 0.07）

    返回：
        loss: scalar，对比损失值
    """
    # 归一化到单位球面上
    sketch_feat = F.normalize(sketch_feat, dim=1)
    image_feat = F.normalize(image_feat, dim=1)

    # 计算相似度矩阵（bs × bs），每行表示一个 sketch 和所有 image 的相似度
    logits_sk2im = torch.matmul(sketch_feat, image_feat.T) / temperature

    # 构造 ground truth，正样本在对角线（即第 i 个 sketch 对应第 i 个 image）
    bs = sketch_feat.size(0)
    labels = torch.arange(bs, device=sketch_feat.device)

    # 计算交叉熵损失（sketch->image 和 image->sketch）
    loss_sk2im = F.cross_entropy(logits_sk2im, labels)

    return loss_sk2im / 2


class EmbeddingSpace(object):
    """
    创建一个特征集合，包含测试集中全部数据
    """
    def __init__(self,
                 img_encoder: Module,
                 skh_img_dataset,
                 skh_img_loader: DataLoader
                 ):
        skh_img_dataset.img()
        img_encoder = img_encoder.eval()

        self.embeddings = []
        self.data_idx = []
        for data in tqdm(skh_img_loader, total=len(skh_img_loader), desc='embeding'):
            images, idxes = data[0].float().cuda(), data[1].long().cuda()

            self.data_idx.append(idxes)

            with torch.no_grad():
                # -> [bs, emb]
                img_embedding = img_encoder(images)

                # show_tensor_map(img_embedding[:, ::10])

                # 获取测试集中全部的图片 embedding，这里embedding的索引即对应它的文件
                self.embeddings.append(img_embedding)

        # -> [all, emb]
        self.embeddings = torch.cat(self.embeddings, dim=0)

        # show_tensor_map(self.embeddings)

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

        # show_tensor_map(distances)

        # -> [bs, k]
        topk_indices = torch.topk(distances, k, largest=False, dim=1)[1]

        searched_idx = self.data_idx[topk_indices]
        return searched_idx


def evaluate(img_encoder, skh_encoder, skh_img_dataset, skh_img_loader):
    skh_img_dataset.img()
    emb_space = EmbeddingSpace(img_encoder, skh_img_dataset, skh_img_loader)

    c_correct_1 = 0
    c_correct_5 = 0
    c_correct_10 = 0
    c_all = 0

    skh_img_dataset.eval()
    skh_encoder = skh_encoder.eval()
    # save_idx = 0
    for idx_batch, data in tqdm(enumerate(skh_img_loader), total=len(skh_img_loader), desc='evaluate'):
        sketch, mask, img, v_index = data[0].float().cuda(), data[1].float().cuda(), data[2].float().cuda(), data[3].long().cuda()

        # for i in range(sketch.size(0)):
        #     c_skh = sketch[i]
        #     vis_s5_data(c_skh.detach().cpu().numpy())

        with torch.no_grad():
            # [bs, emb]
            skh_embedding = skh_encoder(sketch, mask)

            ####################
            # img_embedding = img_encoder(img)
            # skh_embedding = img_embedding

            # vis_tensor_map(skh_embedding[:, ::10], is_show=False, save_root=f'./imgs_gen/sketch_emb_{save_idx}.png')
            # save_idx += 1

            # [bs, k]
            searched_idx = emb_space.top_k(skh_embedding, 10)

            # 计算准确率
            matches = (searched_idx == v_index.unsqueeze(1))  # [bs, k]

            # vis_tensor_map(matches)

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

    dataset = RetrievalDataset(root=data_root, back_mode='S5', max_seq_length=args.max_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)

    '''加载模型及权重'''
    img_encoder = VITFinetune().cuda()
    skh_encoder = SketchTransformer().cuda()

    skh_weight_path = f'./model_trained/retrieval_{args.save_str}.pth'
    img_weight_path = './model_trained/sketch_retrieval_img.pth'
    if args.is_load_weight == 'True':
        try:
            img_encoder.load_state_dict(torch.load(img_weight_path))
            print(Fore.GREEN + 'training image encoder from exist model: ' + img_weight_path)

            skh_encoder.load_state_dict(torch.load(skh_weight_path))
            print(Fore.GREEN + 'training sketch encoder from exist model: ' + skh_weight_path)

        except:
            print(Fore.RED + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        list(skh_encoder.parameters()) + list(img_encoder.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # acc_top1, acc_top5, acc_top10 = evaluate(img_encoder, skh_encoder, dataset, dataloader)

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

            # loss = F.mse_loss(skh_emb, img_emb)
            loss = constructive_loss(skh_emb, img_emb)
            # loss = nt_xent_loss(skh_emb, img_emb)

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

        acc_top1, acc_top5, acc_top10 = evaluate(img_encoder, skh_encoder, dataset, dataloader)
        eval_str = f'{args.save_str}:{epoch}/{args.epoch}-loss: {np.mean(loss_all)} acc_top1: {acc_top1} acc_top5: {acc_top5} acc_top10: {acc_top10}'
        print(eval_str)
        logger.info(eval_str)


if __name__ == '__main__':
    # clear_log('./log')
    # clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    main(parse_args())

