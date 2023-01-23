import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import init
from libs.modeling.FPNTransformer import FPNTrans
from libs.modeling.blocks import LayerNorm


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def random_padding(gt_segments, gt_labels, dt):
    """
    用于填充随机框，填充到100
    dt：特征向量时间维的长度
    """
    padding_segments = []
    padding_labels = []
    for gt_segment, gt_label in zip(gt_segments, gt_labels):
        padding_len = 100 - gt_segment.shape[0]  # 填充到100
        # 起始点位置
        st = torch.randint(0, dt, (padding_len, 1))
        # 动作持续时间，假定100以内均匀分布
        action_len = torch.randint(0, dt, (padding_len, 1))
        padding_box = torch.concat((st, st + action_len), dim=1)
        padding_box.clamp_(max=dt - 1)
        # 随机label
        padding_label = torch.randint(0, 20, (padding_len,)).to(gt_label.dtype)
        padding_box = torch.concat((gt_segment, padding_box), dim=0)
        padding_label = torch.concat((gt_label, padding_label), dim=0)
        padding_segments.append(padding_box)
        padding_labels.append(padding_label)
    return padding_segments, padding_labels


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        # 前两行计算x向量，共64个点
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        # T个时间位置组成频率部分
        pos = torch.arange(T).float()
        # 两两相乘构成T*(d_model//2)的矩阵，并assert形状
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        # 计算不同频率sin, cos值，判断形状，并reshape到T*d_model
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        # MLP层，通过初始编码计算提取特征后的embedding
        # 包含两个线性层，第一个用swish激活函数，第二个不使用激活函数
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


if __name__ == '__main__':
    test_ = FPNTrans()
    noisy_segments = torch.randn(2, 2, 4536)
    noisy_labels = torch.randn(2, 20, 4536)
    fpn_feats = torch.randn(2, 512, 4536)
    fpn_masks = torch.randn(2, 1, 4536).bool()
    ts = torch.Tensor([42, 23]).long()
    noisy_segments, noisy_labels = test_(noisy_segments, noisy_labels, fpn_feats, fpn_masks, ts)
