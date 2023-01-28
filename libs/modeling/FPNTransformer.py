import math
import torch
from torch import nn
from torch.nn import init
from .blocks import MaskedConv1D, get_sinusoid_encoding, LayerNorm, TransformerBlock


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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


class FPNTrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.time_mlp = TimeEmbedding(1000, 2, 512)
        self.dim_list = [0, 2304, 3456, 4032, 4320, 4464, 4536]
        self.head1 = nn.ModuleList()
        self.head2 = nn.ModuleList()
        in_dim1 = [2, 512]
        in_dim2 = [20, 512]
        out_dim = [512, 512]
        kernel_size = 3
        self.embd_norm = []
        for idx in range(2):
            self.head1.append(
                MaskedConv1D(
                    in_dim1[idx], out_dim[idx], kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=False
                )
            )
            self.head2.append(
                MaskedConv1D(
                    in_dim2[idx], out_dim[idx], kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=False
                )
            )
            self.embd_norm.append(LayerNorm(out_dim[idx], device=self.device))
        # # pos_embd: [1, 512, 2304]
        # self.pos_embd = get_sinusoid_encoding(2304, 512).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.stem = nn.ModuleList()
        for idx in range(2):
            self.stem.append(
                TransformerBlock(
                    512, 4,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.01,
                    mha_win_size=19,
                    use_rel_pe=False
                )
            )
        self.tail_lab = nn.ModuleList()
        for idx in range(2):
            self.tail_lab.append(
                TransformerBlock(
                    512, 4,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.01,
                    mha_win_size=19,
                    use_rel_pe=False
                )
            )
        self.tail_seg = nn.ModuleList()
        for idx in range(2):
            self.tail_seg.append(
                TransformerBlock(
                    512, 4,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.01,
                    mha_win_size=19,
                    use_rel_pe=False
                )
            )

    def forward(self, noisy_segments, noisy_labels, fpn_feats, fpn_masks, ts):
        """
        noisy_cls_labels: [2, 20, 4536]
        noisy_offsets: [2, 2, 4536]
        fpn_feats: [2, 512, 4536]
        fpn_masks: [2, 1, 4536]
        """
        # 时间嵌入, time_embd: [2, 512, 1]
        time_embd = self.time_mlp(ts).unsqueeze(-1).to(self.device)
        for idx in range(2):
            noisy_segments, fpn_masks = self.head1[idx](noisy_segments, fpn_masks)
            noisy_labels, fpn_masks = self.head2[idx](noisy_labels, fpn_masks)
            # 嵌入时间信息
            noisy_segments = self.relu(self.embd_norm[idx](noisy_segments + time_embd))
            noisy_labels = self.relu(self.embd_norm[idx](noisy_labels + time_embd))
        # 此处noisy_segments: [2, 512, 4536], noisy_labels: [2, 512, 4536]
        # 暂时不嵌入位置信息
        # 融合两个噪声
        fpn_feats, fpn_masks = self.stem[0](noisy_segments + fpn_feats, fpn_masks)
        fpn_feats, fpn_masks = self.stem[1](noisy_labels + fpn_feats, fpn_masks)
        for idx in range(2):
            noisy_segments, fpn_masks = self.tail_seg[idx](noisy_segments + fpn_feats, fpn_masks)
            noisy_labels, fpn_masks = self.tail_lab[idx](noisy_labels + fpn_feats, fpn_masks)
        return noisy_segments, noisy_labels
