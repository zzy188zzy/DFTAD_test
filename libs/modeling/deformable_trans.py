import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .position_encoding import PositionEmbeddingSine
from .deformable_att import DeformAttn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DeformableTransformer(nn.Module):
    def __init__(self, feature_dimm=2048, nhead=8,
                 num_encoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 activation="relu", num_feature_levels=1, enc_n_points=4
                 ):
        super().__init__()
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=feature_dimm, normalize=True)
        self.level_embed = nn.Parameter(torch.Tensor(1, feature_dimm))
        encoder_layer = DeformableTransformerEncoderLayer(feature_dimm, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # shape=(bs)

    def forward(self, src, mask):
        pos_embed = self.position_embedding(src, mask)
        temporal_lens = []

        bs, c, t = src.shape
        temporal_lens.append(t)
        # (bs, c, t) => (bs, t, c)
        src = src.transpose(1, 2)
        pos_embed = pos_embed.transpose(1, 2)
        lvl_pos_embed = pos_embed + self.level_embed.view(1, 1, -1).to(pos_embed.device)

        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=src.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1,)), temporal_lens.cumsum(0)[:-1]))
        valid_ratios = self.get_valid_ratio(mask)[:, None]

        memory = self.encoder(src, temporal_lens, level_start_index, valid_ratios,
                              lvl_pos_embed, ~mask)
        return memory.transpose(1, 2), mask.unsqueeze(1)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=1, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes,
                                 level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)  # (bs, t)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # (N, t, n_levels)
        return reference_points[..., None]  # (N, t, n_levels, 1)

    def forward(self, src, temporal_lens, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        reference_points = self.get_reference_points(temporal_lens, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
        return output
