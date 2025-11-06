import torch
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1536, dropout=0.1, activation="relu"):
        super().__init__()

        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.h_sigmoid_relu = nn.ReLU6(inplace=True)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # inp = 256
        # oup = 256
        mip = 8
        self.conv1 = nn.Conv2d(d_model, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, d_model, kernel_size=3, stride=1, padding=1)
        self.conv_w = nn.Conv2d(mip, d_model, kernel_size=3, stride=1, padding=1)
        # self.down = nn.Linear(in_features,in_features // 2)
        # self.up = nn.Linear(in_features // 2,in_features)

    def forward(self, tgt):

        identity = tgt

        n, c, h, w = tgt.size()
        # c*1*W
        x_h = self.pool_h(tgt)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(tgt).permute(0, 1, 3, 2)
        # print(x_h.shape)
        y = torch.cat([x_h, x_w], dim=2)
        # print(y.shape) # ([2, 768, 32, 1])

        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = y * (self.h_sigmoid_relu(y+3)/6)
        # print(y.shape) # ([2, 8, 32, 1])

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        tgt = identity * a_w * a_h
        # print(tgt.shape)

        return tgt

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

x = torch.ones(2,768,16,16)
m = DecoderCFALayer(768)
o_v = m(x)
print(o_v.shape)