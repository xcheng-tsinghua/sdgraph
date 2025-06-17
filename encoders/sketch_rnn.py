import math
from typing import Optional, Tuple, Any
import numpy as np
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import einops
from torch.nn import Module
import argparse
import torch.nn.functional as F
from pathlib import Path


def sketch_plot(data):
    data = data.cpu().numpy()
    plt.plot(data[:, 0, 0], data[:, 0, 1])

    plt.show()


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=100, help='batch size in training')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--category', type=str, default='shark', help='diffusion category')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='running on local?')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\quickdraw\raw', help='root of local')
    parser.add_argument('--root_sever', type=str, default=r'/root/my_data/data_set/quickdraw/raw', help='root of sever')

    return parser.parse_args()


def reconstruction_loss(mask: torch.Tensor,
                        target: torch.Tensor,
                        dist: 'BivariateGaussianMixture',
                        q_logits: torch.Tensor):
    # Get $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
    pi, mix = dist.get_distribution()
    # `target` has shape `[seq_len, batch_size, 5]` where the last dimension is the features
    # $(\Delta x, \Delta y, p_1, p_2, p_3)$.
    # We want to get $\Delta x, \Delta$ y and get the probabilities from each of the distributions
    # in the mixture $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
    #
    # `xy` will have shape `[seq_len, batch_size, n_distributions, 2]`
    xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
    # Calculate the probabilities
    # $$p(\Delta x, \Delta y) =
    # \sum_{j=1}^M \Pi_j \mathcal{N} \big( \Delta x, \Delta y \vert
    # \mu_{x,j}, \mu_{y,j}, \sigma_{x,j}, \sigma_{y,j}, \rho_{xy,j}
    # \big)$$
    probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)

    # $$L_s = - \frac{1}{N_{max}} \sum_{i=1}^{N_s} \log \big (p(\Delta x, \Delta y) \big)$$
    # Although `probs` has $N_{max}$ (`longest_seq_len`) elements, the sum is only taken
    # upto $N_s$ because the rest is masked out.
    #
    # It might feel like we should be taking the sum and dividing by $N_s$ and not $N_{max}$,
    # but this will give higher weight for individual predictions in shorter sequences.
    # We give equal weight to each prediction $p(\Delta x, \Delta y)$ when we divide by $N_{max}$
    loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))

    # $$L_p = - \frac{1}{N_{max}} \sum_{i=1}^{N_{max}} \sum_{k=1}^{3} p_{k,i} \log(q_{k,i})$$
    loss_pen = -torch.mean(target[:, :, 2:] * q_logits)

    # $$L_R = L_s + L_p$$
    return loss_stroke + loss_pen


def kl_div_loss(sigma_hat: torch.Tensor, mu: torch.Tensor):
    """
    ## KL-Divergence loss

    This calculates the KL divergence between a given normal distribution and $\mathcal{N}(0, 1)$
    """
    # $$L_{KL} = - \frac{1}{2 N_z} \bigg( 1 + \hat{\sigma} - \mu^2 - \exp(\hat{\sigma}) \bigg)$$
    return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat))


class SketchRNNEmbedding(Module):
    """
    ## Encoder module

    This consists of a bidirectional LSTM
    """

    def __init__(self, enc_hidden_size: int = 256, is_global=True):
        """
        :param enc_hidden_size:
        :param is_global: True: 返回全局特征， False: 返回局部特征
        """
        super().__init__()
        self.is_global = is_global
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)

        if self.is_global:
            self.mu_head = nn.Linear(2 * enc_hidden_size, 2 * enc_hidden_size)
        else:
            self.mu_head = nn.Conv1d(2 * enc_hidden_size, 2 * enc_hidden_size, 1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor,  state=None):
        """
        :param inputs: [bs, len, emb]
        :param mask: [bs, len]
        :param state:
        :return:
        """
        inputs = inputs.transpose(0, 1)

        # -> output: [n_pnt, bs, channel]
        output, (hidden, cell) = self.lstm(inputs.float(), state)

        if self.is_global:
            hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')  # fb: forward backward
        else:
            hidden = output.permute(1, 2, 0)

        emb = self.mu_head(hidden)
        return emb


class SketchRNN_Cls(Module):
    def __init__(self, n_classes: int, dropout=0.4):
        super().__init__()
        print('create sketch_rnn classifier')

        self.encoder = SketchRNNEmbedding()

        channel_l0 = 512
        channel_l1 = int((channel_l0 * n_classes) ** 0.5)
        channel_l2 = n_classes

        self.linear = nn.Sequential(
            nn.BatchNorm1d(channel_l0),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(channel_l0, channel_l1),
            nn.BatchNorm1d(channel_l1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(channel_l1, channel_l2),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor=None):
        fea = self.encoder(inputs, mask)
        fea = self.linear(fea)

        fea = F.log_softmax(fea, dim=1)

        return fea


class SketchRNN_Seg(Module):
    def __init__(self, n_classes: int, dropout=0.4):
        super().__init__()
        print('create sketch_rnn classifier')

        self.encoder = SketchRNNEmbedding(is_global=False)

        channel_l0 = 512
        channel_l1 = int((channel_l0 * n_classes) ** 0.5)
        channel_l2 = n_classes

        self.linear = nn.Sequential(
            nn.BatchNorm1d(channel_l0),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(channel_l0, channel_l1, 1),
            nn.BatchNorm1d(channel_l1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(channel_l1, channel_l2, 1),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor=None):
        fea = self.encoder(inputs, mask)  # -> [bs, channel, n_pnts]
        fea = self.linear(fea)  # -> [bs, channel, n_pnts]

        fea = F.log_softmax(fea, dim=1)
        return fea


class StrokesDataset(Dataset):
    """
    ## Dataset

    This class loads and pre-processes the data.
    """
    def __init__(self, root_npz: str, tag: str, max_seq_length: int, scale: Optional[float] = None):
        """
        `dataset` is a list of numpy arrays of shape [seq_len, 3].
        It is a sequence of strokes, and each stroke is represented by
        3 integers.
        First two are the displacements along x and y ($\Delta x$, $\Delta y$)
        and the last integer represents the state of the pen, $1$ if it's touching
        the paper and $0$ otherwise.
        """

        dataset = np.load(str(root_npz), encoding='latin1', allow_pickle=True)[tag]

        data = []  # 其中包含全部草图，不止一张草图
        for seq in dataset:
            # 筛选点数符合要求的草图，点数必须在(10, max_seq_length]之间
            if 10 < len(seq) <= max_seq_length:
                # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                # Convert to a floating point array and add to `data`
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        # We then calculate the scaling factor which is the
        # standard deviation of ($\Delta x$, $\Delta y$) combined.
        # Paper notes that the mean is not adjusted for simplicity,
        # since the mean is anyway close to $0$.
        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        # Get the longest sequence length among all sequences
        longest_seq_len = max([len(seq) for seq in data])

        # We initialize PyTorch data array with two extra steps for start-of-sequence (sos)
        # and end-of-sequence (eos).
        # Each step is a vector $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # Only one of $p_1, p_2, p_3$ is $1$ and the others are $0$.
        # They represent *pen down*, *pen up* and *end-of-sequence* in that order.
        # $p_1$ is $1$ if the pen touches the paper in the next step.
        # $p_2$ is $1$ if the pen doesn't touch the paper in the next step.
        # $p_3$ is $1$ if it is the end of the drawing.
        # dim0: 数据集中的草图数
        # dim1: 草图中的点数 + 2
        # dim2: x, y, p1, p2, p3
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)  # =2 是因为新增了草图开始和草图结束两个额外步骤

        # The mask array needs only one extra-step since it is for the outputs of the
        # decoder, which takes in `data[:-1]` and predicts next step.
        # 即哪些点是有效的，因为不同草图中的点不同，数值为1及=即该位置的点是有效的，
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            # 设置点坐标，注意避开第一行及最后一行
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
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


class BivariateGaussianMixture(object):
    """
    ## Bi-variate Gaussian mixture

    The mixture is represented by $\Pi$ and
    $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
    This class adjusts temperatures and creates the categorical and Gaussian
    distributions from the parameters.
    """
    def __init__(self, pi_logits: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor,
                 sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor):
        super().__init__()
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_distributions(self):
        """Number of distributions in the mixture, $M$"""
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        """
        Adjust by temperature $\tau$
        """
        # $$\hat{\Pi_k} \leftarrow \frac{\hat{\Pi_k}}{\tau}$$
        self.pi_logits /= temperature
        # $$\sigma^2_x \leftarrow \sigma^2_x \tau$$
        self.sigma_x *= math.sqrt(temperature)
        # $$\sigma^2_y \leftarrow \sigma^2_y \tau$$
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        # Clamp $\sigma_x$, $\sigma_y$ and $\rho_{xy}$ to avoid getting `NaN`s
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)

        # Get means
        mean = torch.stack([self.mu_x, self.mu_y], -1)
        # Get covariance matrix
        cov = torch.stack([
            sigma_x * sigma_x, rho_xy * sigma_x * sigma_y,
            rho_xy * sigma_x * sigma_y, sigma_y * sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)

        # Create bi-variate normal distribution.
        #
        # It would be efficient to `scale_tril` matrix as `[[a, 0], [b, c]]`
        # where
        # $$a = \sigma_x, b = \rho_{xy} \sigma_y, c = \sigma_y \sqrt{1 - \rho^2_{xy}}$$.
        # But for simplicity we use co-variance matrix.
        # [This is a good resource](https://www2.stat.duke.edu/courses/Spring12/sta104.1/Lectures/Lec22.pdf)
        # if you want to read up more about bi-variate distributions, their co-variance matrix,
        # and probability density function.
        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

        # Create categorical distribution $\Pi$ from logits
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        return cat_dist, multi_dist


class EncoderRNN(Module):
    """
    ## Encoder module

    This consists of a bidirectional LSTM
    """

    def __init__(self, d_z: int = 128, enc_hidden_size: int = 256):
        super().__init__()
        # Create a bidirectional LSTM taking a sequence of
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$ as input.
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        # Head to get $\mu$
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        # Head to get $\hat{\sigma}$
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)

    def forward(self, inputs: torch.Tensor, state=None):
        # The hidden state of the bidirectional LSTM is the concatenation of the
        # output of the last token in the forward direction and
        # first token in the reverse direction, which is what we want.
        # $$h_{\rightarrow} = encode_{\rightarrow}(S),
        # h_{\leftarrow} = encode←_{\leftarrow}(S_{reverse}),
        # h = [h_{\rightarrow}; h_{\leftarrow}]$$
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        output = self.lstm(inputs.float(), state)

        # The state has shape `[2, batch_size, hidden_size]`,
        # where the first dimension is the direction.
        # We rearrange it to get $h = [h_{\rightarrow}; h_{\leftarrow}]$
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')

        # $\mu$
        mu = self.mu_head(hidden)
        # $\hat{\sigma}$
        sigma_hat = self.sigma_head(hidden)
        # $\sigma = \exp(\frac{\hat{\sigma}}{2})$
        sigma = torch.exp(sigma_hat / 2.)

        # Sample $z = \mu + \sigma \cdot \mathcal{N}(0, I)$
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))

        return z, mu, sigma_hat


class DecoderRNN(Module):
    """
    ## Decoder module

    This consists of a LSTM
    """
    def __init__(self, d_z: int = 128, dec_hidden_size: int = 512, n_distributions: int = 20):
        super().__init__()
        # LSTM takes $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ as input
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)

        # Initial state of the LSTM is $[h_0; c_0] = \tanh(W_{z}z + b_z)$.
        # `init_state` is the linear transformation for this
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)

        # This layer produces outputs for each of the `n_distributions`.
        # Each distribution needs six parameters
        # $(\hat{\Pi_i}, \mu_{x_i}, \mu_{y_i}, \hat{\sigma_{x_i}}, \hat{\sigma_{y_i}} \hat{\rho_{xy_i}})$
        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_distributions)

        # This head is for the logits $(\hat{q_1}, \hat{q_2}, \hat{q_3})$
        self.q_head = nn.Linear(dec_hidden_size, 3)
        # This is to calculate $\log(q_k)$ where
        # $$q_k = \operatorname{softmax}(\hat{q})_k = \frac{\exp(\hat{q_k})}{\sum_{j = 1}^3 \exp(\hat{q_j})}$$
        self.q_log_softmax = nn.LogSoftmax(-1)

        # These parameters are stored for future reference
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def forward(self, x: torch.Tensor, z: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # Calculate the initial state
        if state is None:
            # $[h_0; c_0] = \tanh(W_{z}z + b_z)$
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            # `h` and `c` have shapes `[batch_size, lstm_size]`. We want to shape them
            # to `[1, batch_size, lstm_size]` because that's the shape used in LSTM.
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())

        # Run the LSTM
        outputs, state = self.lstm(x, state)

        # Get $\log(q)$
        q_logits = self.q_log_softmax(self.q_head(outputs))

        # Get $(\hat{\Pi_i}, \mu_{x,i}, \mu_{y,i}, \hat{\sigma_{x,i}},
        # \hat{\sigma_{y,i}} \hat{\rho_{xy,i}})$.
        # `torch.split` splits the output into 6 tensors of size `self.n_distribution`
        # across dimension `2`.
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixtures(outputs), self.n_distributions, 2)

        # Create a bi-variate Gaussian mixture
        # $\Pi$ and
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        # where
        # $$\sigma_{x,i} = \exp(\hat{\sigma_{x,i}}), \sigma_{y,i} = \exp(\hat{\sigma_{y,i}}),
        # \rho_{xy,i} = \tanh(\hat{\rho_{xy,i}})$$
        # and
        # $$\Pi_i = \operatorname{softmax}(\hat{\Pi})_i = \frac{\exp(\hat{\Pi_i})}{\sum_{j = 1}^3 \exp(\hat{\Pi_j})}$$
        #
        # $\Pi$ is the categorical probabilities of choosing the distribution out of the mixture
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y,
                                        torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        return dist, q_logits, state


class Sampler(object):
    """
    ## Sampler

    This samples a sketch from the decoder and plots it
    """
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def sample(self, data: torch.Tensor, category: str, plot_idx: int, temperature: float = 0.4):
        # $N_{max}$
        longest_seq_len = len(data)

        # sketch_plot(data)

        # Get $z$ from the encoder
        z, _, _ = self.encoder(data)

        # Start-of-sequence stroke is $(0, 0, 1, 0, 0)$
        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]
        # Initial decoder is `None`.
        # The decoder will initialize it to $[h_0; c_0] = \tanh(W_{z}z + b_z)$
        state = None

        # We don't need gradients
        with torch.no_grad():
            # Sample $N_{max}$ strokes
            for i in range(longest_seq_len):
                # $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ is the input to the decoder
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                # Get $\Pi$, $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$,
                # $q$ and the next state from the decoder
                dist, q_logits, state = self.decoder(data, z, state)
                # Sample a stroke
                s = self._sample_step(dist, q_logits, temperature)
                # Add the new stroke to the sequence of strokes
                seq.append(s)
                # Stop sampling if $p_3 = 1$. This indicates that sketching has stopped
                if s[4] == 1:
                    break

        # Create a PyTorch tensor of the sequence of strokes
        seq = torch.stack(seq)

        # Plot the sequence of strokes
        self.plot(seq, category, plot_idx)

    @staticmethod
    def _sample_step(dist: 'BivariateGaussianMixture', q_logits: torch.Tensor, temperature: float):
        # Set temperature $\tau$ for sampling. This is implemented in class `BivariateGaussianMixture`.
        dist.set_temperature(temperature)
        # Get temperature adjusted $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        pi, mix = dist.get_distribution()
        # Sample from $\Pi$ the index of the distribution to use from the mixture
        idx = pi.sample()[0, 0]

        # Create categorical distribution $q$ with log-probabilities `q_logits` or $\hat{q}$
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        # Sample from $q$
        q_idx = q.sample()[0, 0]

        # Sample from the normal distributions in the mixture and pick the one indexed by `idx`
        xy = mix.sample()[0, 0, idx]

        # Create an empty stroke $(\Delta x, \Delta y, q_1, q_2, q_3)$
        stroke = q_logits.new_zeros(5)
        # Set $\Delta x, \Delta y$
        stroke[:2] = xy
        # Set $q_1, q_2, q_3$
        stroke[q_idx + 2] = 1
        #
        return stroke

    @staticmethod
    def plot(seq: torch.Tensor, category: str, plot_idx: int):
        # Take the cumulative sums of $(\Delta x, \Delta y)$ to get $(x, y)$
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        # Create a new numpy array of the form $(x, y, q_2)$
        seq[:, 2] = seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()

        # Split the array at points where $q_2$ is $1$.
        # i.e. split the array of strokes at the points where the pen is lifted from the paper.
        # This gives a list of sequence of strokes.
        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
        # Plot each sequence of strokes
        plt.clf()
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        # Don't show axes
        plt.axis('off')
        # Show the plot

        # global gen_idx
        c_root = Path(__file__).resolve()
        parent_dir = c_root.parent.parent
        save_dir = os.path.join(parent_dir, 'imgs_gen', 'sketchrnn', category)
        os.makedirs(save_dir, exist_ok=True)
        save_root = os.path.join(save_dir, f'{plot_idx}.png')

        # save image
        plt.savefig(save_root)


def main():
    args = parse_args()

    if args.local == 'True':
        root = args.root_local
    else:
        root = args.root_sever

    npz_root = os.path.join(root, f'{args.category}.full.npz')
    print(f'loading npz file from: {npz_root}')

    '''定义数据集'''
    train_dataset = StrokesDataset(npz_root, 'train', 200)
    valid_dataset = StrokesDataset(npz_root, 'valid', 200, train_dataset.scale)

    train_loader = DataLoader(train_dataset, args.bs, shuffle=True)

    '''定义模型'''
    encoder = EncoderRNN().cuda()
    decoder = DecoderRNN().cuda()
    sampler = Sampler(encoder, decoder)

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

        for batch_id, batch in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            optimizer.zero_grad()

            data = batch[0].transpose(0, 1).cuda()
            mask = batch[1].transpose(0, 1).cuda()

            z, mu, sigma_hat = encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
            inputs = torch.cat([data[:-1], z_stack], 2)
            dist, q_logits, _ = decoder(inputs, z, None)

            kl_loss = kl_div_loss(sigma_hat, mu)
            rect_loss = reconstruction_loss(mask, data[1:], dist, q_logits)
            loss = rect_loss + 0.5 * kl_loss

            loss.backward()

            # 防止梯度爆炸，必须放在 loss.backward() 之后，optimizer.step() 之前
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            optimizer.step()

        scheduler.step()

        '''生成草图'''
        with torch.no_grad():
            encoder = encoder.eval()
            decoder = decoder.eval()

            # Randomly pick a sample from validation dataset to encoder
            data, *_ = valid_dataset[np.random.choice(len(valid_dataset))]

            # Add batch dimension and move it to device
            data = data.unsqueeze(1).cuda()

            # Sample
            sampler.sample(data, args.category, skh_gen_idx)
            skh_gen_idx += 1


if __name__ == "__main__":
    main()



