from encoders_3rd import sketch_rnn
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from data_utils import sketch_utils
import time


class SketchProjDataset(Dataset):
    def __init__(self, file_root: str, max_seq_length: int):
        """
        为草图项目设计的数据集读取类
        :param file_root:
        :param max_seq_length:
        """
        file_all = sketch_utils.get_allfiles(file_root)
        print(f'txt file all: {len(file_all)}, from: ', file_root)
        data = []

        for c_file in file_all:
            c_np = np.loadtxt(c_file, delimiter=',')

            if 10 < len(c_np) <= max_seq_length:

                # 归一化到 [-1, 1]
                c_np = sketch_utils.sketch_std(c_np)

                # 转化为相对坐标
                c_np[1:, :2] = c_np[1:, :2] - c_np[:-1, :2]

                data.append(c_np)

        longest_seq_len = max([len(seq) for seq in data])
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)  # =2 是因为新增了草图开始和草图结束两个额外步骤

        # mask 即哪些点是有效的，因为不同草图中的点不同，数值为1及=即该位置的点是有效的，
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            # 设置点坐标，注意避开第一行及最后一行
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2]
            # $p_1$
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            # $p_2$
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            # $p_3$
            # 草图最后一个点的状态设为1，即草图结束
            self.data[i, len_seq + 1:, 4] = 1
            # Mask is on until end of sequence
            # 设定哪些点是有效的
            self.mask[i, :len_seq + 1] = 1

        # Start-of-sequence is $(0, 0, 1, 0, 0)$
        self.data[:, 0, 2] = 1

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample"""
        return self.data[idx], self.mask[idx]


def parse_args_sketch_proj():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=15, help='batch size in training')
    parser.add_argument('--epoch', default=100000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_skh_gen', default=1000, type=int, help='---')

    parser.add_argument('--category', type=str, default='shark', help='diffusion category')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='running on local?')
    parser.add_argument('--root_local', type=str, default=r'C:\Users\ChengXi\Desktop\sketchrnn_proj_txt',
                        help='root of local')
    parser.add_argument('--root_sever', type=str, default=r'/root/my_data/data_set/sketch_cad/sketchrnn_proj_txt',
                        help='root of sever')

    return parser.parse_args()


def train_sketch_rnn_proj():
    """
    草图项目的草图补全
    :return:
    """
    args = parse_args_sketch_proj()

    if args.local == 'True':
        root = args.root_local
    else:
        root = args.root_sever

    '''定义数据集'''
    train_dataset = SketchProjDataset(root, 200)
    train_loader = DataLoader(train_dataset, args.bs, shuffle=True)

    '''定义模型'''
    predictor = sketch_rnn.SketchRNN().cuda()

    model_save = './model_trained/sketchrnn_proj.pth'
    model_save = os.path.abspath(model_save)

    try:
        predictor.load_state_dict(torch.load(model_save))
        print('training from exist model: ' + model_save)
    except:
        print('no existing model, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        params=predictor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    '''训练'''
    for epoch in range(args.epoch):
        predictor = predictor.train()
        epoch_loss = []

        for batch_id, batch in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'epoch: {epoch} / {args.epoch}'):
            optimizer.zero_grad()

            data = batch[0].transpose(0, 1).cuda()
            mask = batch[1].transpose(0, 1).cuda()

            sigma_hat, mu, dist, q_logits = predictor(data)

            kl_loss = sketch_rnn.kl_div_loss(sigma_hat, mu)
            rect_loss = sketch_rnn.reconstruction_loss(mask, data[1:], dist, q_logits)
            loss = rect_loss + 0.5 * kl_loss

            loss.backward()
            epoch_loss.append(loss.item())

            # 防止梯度爆炸，必须放在 loss.backward() 之后，optimizer.step() 之前
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

            optimizer.step()

        print(f'loss: {np.mean(epoch_loss).item()}')
        scheduler.step()

        torch.save(predictor.state_dict(), model_save)
        print('save weight to ', model_save)

        time.sleep(2)


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=500, help='batch size in training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_skh_gen', default=1000, type=int, help='---')

    parser.add_argument('--category', type=str, default='shark', help='diffusion category')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='running on local?')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\quickdraw\raw', help='root of local')
    parser.add_argument('--root_sever', type=str, default=r'/root/my_data/data_set/quickdraw/raw', help='root of sever')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.local == 'True':
        root = args.root_local
    else:
        root = args.root_sever

    # npz_root = os.path.join(root, f'{args.category}.full.npz')
    npz_root = os.path.join(root, f'{args.category}.full.npz')
    print(f'loading npz file from: {npz_root}')

    '''定义数据集'''
    train_dataset = sketch_rnn.StrokesDataset(npz_root, 'train', 200)
    valid_dataset = sketch_rnn.StrokesDataset(npz_root, 'valid', 200, train_dataset.scale)

    train_loader = DataLoader(train_dataset, args.bs, shuffle=True)

    '''定义模型'''
    encoder = sketch_rnn.EncoderRNN().cuda()
    decoder = sketch_rnn.DecoderRNN().cuda()
    # sampler = Sampler(encoder, decoder)

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        params=list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    skh_gen_idx = 0

    '''训练'''
    for epoch in range(args.epoch):
        encoder = encoder.train()
        decoder = decoder.train()

        for batch_id, batch in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'epoch: {epoch} / {args.epoch}'):
            optimizer.zero_grad()

            data = batch[0].transpose(0, 1).cuda()
            mask = batch[1].transpose(0, 1).cuda()

            z, mu, sigma_hat = encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
            inputs = torch.cat([data[:-1], z_stack], 2)
            dist, q_logits, _ = decoder(inputs, z, None)

            kl_loss = sketch_rnn.kl_div_loss(sigma_hat, mu)
            rect_loss = sketch_rnn.reconstruction_loss(mask, data[1:], dist, q_logits)
            loss = rect_loss + 0.5 * kl_loss

            loss.backward()

            # 防止梯度爆炸，必须放在 loss.backward() 之后，optimizer.step() 之前
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            optimizer.step()

        scheduler.step()

    '''生成草图'''
    with torch.no_grad():

        for _ in tqdm(range(args.n_skh_gen), total=args.n_skh_gen, desc='generate sketches'):
            encoder = encoder.eval()
            decoder = decoder.eval()
            sampler = sketch_rnn.Sampler(encoder, decoder)

            # Randomly pick a sample from validation dataset to encoder
            data, *_ = valid_dataset[np.random.choice(len(valid_dataset))]

            # Add batch dimension and move it to device
            data = data.unsqueeze(1).cuda()

            # Sample
            sampler.sample(data, args.category, skh_gen_idx)
            skh_gen_idx += 1


if __name__ == '__main__':
    train_sketch_rnn_proj()
    # main()



