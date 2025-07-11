"""
Multi-Graph Transformer
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feedforward_dim=512, dropout=0.1):
        super().__init__()
        self.sub_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU()
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters
        # with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim=None,
                 val_dim=None, key_dim=None, dropout=0.1):
        super().__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Args:
            q: Input queries (batch_size, n_query, input_dim)
            h: Input data (batch_size, graph_size, input_dim)
            mask: Input attention mask (batch_size, n_query, graph_size)
                  or viewable as that (i.e. can be 2 dim if n_query == 1);
                  Mask should contain -inf if attention is not possible
                  (i.e. mask is a negative adjacency matrix)

        Returns:
            out: Updated data after attention (batch_size, graph_size, input_dim)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)

        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # out = self.drop(out)

        return out


class SkipConnection(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class MultiGraphTransformerLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim,
                 normalization='batch', dropout=0.1):
        super().__init__()

        self.self_attention1 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )
        self.self_attention2 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )

        self.self_attention3 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )
        # modified on 2019 10 26.
        self.tmp_linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim, bias=True),
            nn.ReLU(),
        )

        self.norm1 = Normalization(embed_dim, normalization)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, input, mask1, mask2, mask3):
        # ipdb.set_trace()
        h1 = self.self_attention1(input, mask=mask1)
        h2 = self.self_attention2(input, mask=mask2)
        h3 = self.self_attention3(input, mask=mask3)
        hh = torch.cat((h1, h2, h3), dim=2)
        hh = self.tmp_linear_layer(hh)
        # ipdb.set_trace()
        hh = self.norm1(hh, mask=mask1)
        hh = self.positionwise_ff(hh, mask=mask1)
        hh = self.norm2(hh, mask=mask1)
        return hh


class GraphTransformerEncoder(nn.Module):

    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=6, n_heads=8,
                 embed_dim=512, feedforward_dim=2048, normalization='batch', dropout=0.1):
        super().__init__()

        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)

        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            MultiGraphTransformerLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask1=None, attention_mask2=None, attention_mask3=None):

        coor_emb = self.coord_embed(coord)
        flag_emb = self.feat_embed(flag)

        h = torch.cat((coor_emb, flag_emb), dim=2)
        h = torch.cat((h, self.feat_embed(pos)), dim=2)

        # Perform n_layers of Graph Transformer blocks
        for layer in self.transformer_layers:
            h = layer(h, mask1=attention_mask1, mask2=attention_mask2, mask3=attention_mask3)

        return h


class MGT(nn.Module):
    def __init__(self, n_classes=345, coord_input_dim=2, feat_input_dim=2, feat_dict_size=103,
                 n_layers=4, n_heads=8, embed_dim=256, feedforward_dim=1024,
                 normalization='batch', dropout=0.25, mlp_classifier_dropout=0.25):
        super().__init__()

        print('Create Multi-Graph Transformer classification network.')
        self.encoder = GraphTransformerEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, n_layers,
            n_heads, embed_dim, feedforward_dim, normalization, dropout)

        self.mlp_classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )

    def forward(self, coord, additional):
        """
        :param coord: xy, [bs, 100, 2]
        :param additional:
        :return:
        """
        flag, pos, attention_mask1, attention_mask2, attention_mask3, padding_mask = additional
        # flag: state, [bs, 100], 100: pen_down, 101: pen_up, 102: padding
        # pos: position, [bs, 100]
        # attention_mask1:
        # attention_mask2:
        # attention_mask3:
        # padding_mask:

        # Embed input sequence
        h = self.encoder(coord, flag, pos, attention_mask1, attention_mask2, attention_mask3)

        # Mask out padding embeddings to zero
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)

        else:
            g = h.sum(dim=1)

        # Compute logits
        logits = self.mlp_classifier(g)
        logits = F.log_softmax(logits, dim=1)

        return logits


def produce_adjacent_matrix_2_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    adja_matr = np.zeros([100, 100], int)

    adja_matr[:][:] = -1e10

    adja_matr[0][0] = 0
    if (flag_bits[0] == 100):
        adja_matr[0][1] = 0

    for idx in range(1, stroke_len):
        #
        adja_matr[idx][idx] = 0

        if (flag_bits[idx - 1] == 100):
            adja_matr[idx][idx - 1] = 0

        if idx == stroke_len - 1:
            break

        if (flag_bits[idx] == 100):
            adja_matr[idx][idx + 1] = 0

    return adja_matr


def produce_adjacent_matrix_4_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    adja_matr = np.zeros([100, 100], int)
    adja_matr[:][:] = -1e10

    adja_matr[0][0] = 0
    if (flag_bits[0] == 100):
        adja_matr[0][1] = 0
        #
        if (flag_bits[1] == 100):
            adja_matr[0][2] = 0

    for idx in range(1, stroke_len):
        #
        adja_matr[idx][idx] = 0

        if (flag_bits[idx - 1] == 100):
            adja_matr[idx][idx - 1] = 0
            #
            if (idx >= 2) and (flag_bits[idx - 2] == 100):
                adja_matr[idx][idx - 2] = 0

        if idx == stroke_len - 1:
            break

        #
        if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 100):
            adja_matr[idx][idx + 1] = 0
            #
            if (idx <= (stroke_len - 3)) and (flag_bits[idx + 1] == 100):
                adja_matr[idx][idx + 2] = 0

    return adja_matr


def produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_len):
    assert flag_bits.shape == (100, 1)
    adja_matr = np.zeros([100, 100], int)
    adja_matr[:][:] = -1e10

    adja_matr[0][0] = 0
    adja_matr[0][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][0] = 0

    assert flag_bits[0] == 100 or flag_bits[0] == 101

    if (flag_bits[0] == 101) and stroke_len >= 2:
        adja_matr[0][1] = 0

    for idx in range(1, stroke_len):

        assert flag_bits[idx] == 100 or flag_bits[idx] == 101

        adja_matr[idx][idx] = 0

        if (flag_bits[idx - 1] == 101):
            adja_matr[idx][idx - 1] = 0

        if (idx == stroke_len - 1):
            break

        #
        if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 101):
            adja_matr[idx][idx + 1] = 0

    return adja_matr


def generate_padding_mask(stroke_length):
    padding_mask = np.ones([100, 1], int)
    padding_mask[stroke_length:, :] = 0
    return padding_mask


def test():
    # bs = 3
    # length = 100
    # xy = torch.rand(bs, length, 2).cuda()
    # flag = torch.randint(100, 103, (bs, length)).cuda()
    # pos = torch.arange(100).repeat(bs, 1).cuda()
    #
    # # attention_mask1 = torch.from_numpy(produce_adjacent_matrix_2_neighbors(flag[0][:, np.newaxis], 50)).repeat(bs, 1, 1).cuda()
    # # attention_mask2 = torch.from_numpy(produce_adjacent_matrix_4_neighbors(flag[0][:, np.newaxis], 50)).repeat(bs, 1, 1).cuda()
    # # attention_mask3 = torch.from_numpy(produce_adjacent_matrix_joint_neighbors(flag[0][:, np.newaxis], 50)).repeat(bs, 1, 1).cuda()
    # # padding_mask = torch.from_numpy(generate_padding_mask(50)).repeat(bs, 1).cuda()
    #
    # anet = MGT().cuda()
    #
    # res = anet(xy, (flag, pos, None, None, None, None))
    # print(res.size())

    import pickle
    import matplotlib.pyplot as plt

    apic = r'E:\document\DeepLearning\multigraph_transformer\dataloader\tiny_train_dataset_dict.pickle'
    pic_file = open(apic, 'rb')
    apicval = pickle.load(pic_file)

    coordinate, flag_bits, stroke_len = apicval['/home/peng/dataset/tiny_quickdraw_coordinate/train/The_Eiffel_Tower/The_Eiffel_Tower_0.npy']

    plt.plot(coordinate[:, 0], coordinate[:, 1])
    plt.show()

    attention_mask_2_neighbors = produce_adjacent_matrix_2_neighbors(flag_bits, stroke_len)

    attention_mask_4_neighbors = produce_adjacent_matrix_4_neighbors(flag_bits, stroke_len)

    attention_mask_joint_neighbors = produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_len)

    padding_mask = generate_padding_mask(stroke_len)

    asliar = 0
    pass


if __name__ == '__main__':
    test()
    pass

