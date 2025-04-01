from torch import nn
import torch

# 预训练的pointbert对于每个输入的点云，其输出的向量维度
POINTBERT_CHANNEL_OUT = 512


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class Group(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_group = 512
        self.group_size = 32

    def forward(self, xyz):
        '''
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1)
        )

    def forward(self, point_groups):
        '''
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)

        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]

        return feature_global.reshape(bs, g, 256)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 6
        self.scale = 0.125

        self.qkv = nn.Linear(384, 384 * 3, bias=False)
        self.proj = nn.Linear(384, 384)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384)
        self.norm2 = nn.LayerNorm(384)

        self.mlp = Mlp()
        self.attn = Attention()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """
    transformer Encoder without hierarchical structure
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.group_divider = Group()
        self.encoder = Encoder()
        self.reduce_dim = nn.Linear(256, 384)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, 384))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )

        self.blocks = TransformerEncoder()
        self.norm = nn.LayerNorm(384)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        return concat_f


class PointBERT_ULIP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = PointTransformer()
        self.pc_projection = nn.Parameter(torch.empty(768, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    @torch.inference_mode()
    def forward(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    @property
    def channel_out(self):
        """
        将一个点云生成的对应的向量长度
        :return: 512
        """
        return POINTBERT_CHANNEL_OUT


def create_pretrained_pointbert(root_ckpt: str = './model_trained/pointbert_ulip2.pth'):
    print('create pretrained pointBERT, load weight from ' + root_ckpt)

    pointbert_pretrained = PointBERT_ULIP2()

    try:
        pointbert_pretrained.load_state_dict(torch.load(root_ckpt), strict=True)
    except:
        raise ValueError('can not load pretrained model weight: ', root_ckpt)

    # torch.save(pointbert_pretrained.state_dict(), root_ckpt)

    # 设为评估模式
    pointbert_pretrained = pointbert_pretrained.eval()

    # 禁用梯度计算，提升速度
    pointbert_pretrained.requires_grad_(False)

    return pointbert_pretrained


