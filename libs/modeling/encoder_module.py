import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        time_dim = 2304
        self.relu = nn.ReLU(inplace=True)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        self.embd.append(
            MaskedConv1D(
                2, 256, 3,
                stride=1, padding=3 // 2, bias=False
            )
        )
        self.embd.append(
            MaskedConv1D(
                256, 512, 3,
                stride=1, padding=3 // 2, bias=False
            )
        )
        self.embd_norm.append(LayerNorm(256))
        self.embd_norm.append(LayerNorm(512))
        self.stem = nn.ModuleList()
        for idx in range(2):
            self.stem.append(
                TransformerBlock(
                    512, 4,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    mha_win_size=19,
                    use_rel_pe=False
                )
            )

    def forward(self, mask, noisy_labels, t):
        time_ = self.time_mlp(t)
        noisy_labels = noisy_labels.transpose(1, 2)
        # embedding network
        for idx in range(len(self.embd)):
            noisy_labels, mask = self.embd[idx](noisy_labels, mask)
            noisy_labels = noisy_labels + time_
            noisy_labels = self.relu(self.embd_norm[idx](noisy_labels))
        # stem transformer
        for idx in range(len(self.stem)):
            noisy_labels, mask = self.stem[idx](noisy_labels, mask)
        return noisy_labels


if __name__ == '__main__':
    t = torch.randint(0, 300, (2,)).long()
    noisy_labels = torch.randn(2, 2304, 2)
    mask = torch.ones([2, 2304], dtype=torch.bool)
    mask[0, 1330:] = False
    mask[0, 989:] = False
    model = TransformerEncoder()
    model(mask, noisy_labels, t)
