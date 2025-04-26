"""
训练检索
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from torch.nn import Module
import torch.nn.functional as F
import argparse
from encoders.vit import create_pretrained_VIT
from data_utils.sketch_dataset import RetrievalDataset
from encoders.sketch_rnn import SketchRNNEmbedding as SketchEncoder
from tqdm import tqdm
from colorama import Fore, Back, init
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--save_str', type=str, default='sdgraph_unet', help='---')

    parser.add_argument('--bs', type=int, default=200, help='batch size in training')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=f'/root/my_data/data_set/sketch_cad/', help='root of dataset')
    parser.add_argument('--root_local', type=str, default=fr'D:\document\DeepLearning\DataSet\sketch_cad\raw', help='root of dataset')

    '''
    parser.add_argument('--root_sever', type=str, default=f'/root/my_data/data_set/unified_sketch_from_quickdraw/stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}',  help='root of dataset')
    parser.add_argument('--root_local', type=str, default=f'D:/document/DeepLearning/DataSet/unified_sketch_from_quickdraw/apple_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}', help='root of dataset')
    '''

    return parser.parse_args()


class ContrastiveLoss(Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, label: torch.Tensor):
        if outputs.size()[0] == 2:
            euclidean_distance = F.pairwise_distance(outputs[0], outputs[1], keepdim=True)

            loss_contrastive = torch.mean(
                (1 - label) * torch.pow(euclidean_distance, 2) +
                label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            )

            return loss_contrastive

        raise ValueError('This is the Contrastive Loss but more (or less) than 2 tensors were unpacked')


class TripletLoss(Module):
    def __init__(self, margin: int):
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 3:
            positive_distance = F.pairwise_distance(outputs[0], outputs[1])
            negative_distance = F.pairwise_distance(outputs[0], outputs[2])
            losses = torch.relu(positive_distance - negative_distance + self.margin)

            return torch.mean(losses)

        raise ValueError('This is the Triplet Loss but more (or less) than 3 tensors were unpacked')


def main(args):

    # 设置数据集
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = RetrievalDataset(root=data_root, mode='train', return_mode='S5')
    test_dataset = RetrievalDataset(root=data_root, mode='test', return_mode='S5')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    '''加载模型及权重'''
    image_encoder = create_pretrained_VIT().cuda()
    sketch_encoder = SketchEncoder().cuda()

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        sketch_encoder.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    for epoch in range(args.epoch):
        sketch_encoder = sketch_encoder.train()
        loss_all = []

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            img, text, mask = data[0].float().cuda(), data[1].float().cuda(), data[2].float().cuda()

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            sketch_emb = sketch_encoder(text, mask)
            img_emb = image_encoder(img).detach().clone()

            loss = F.mse_loss(sketch_emb, img_emb)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())

        scheduler.step()
        torch.save(sketch_encoder.state_dict(), './model_trained/sketch_retrieval.pth')
        print(Back.BLUE + f'{epoch} / {args.epoch}: save sketch weights at: ./weights/sketch_encoder.pth, loss: {np.mean(loss_all)}')


if __name__ == '__main__':
    main(parse_args())


