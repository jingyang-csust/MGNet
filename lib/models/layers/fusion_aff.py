import torch
import torch.nn as nn
from timm.models.layers import Mlp
from torch.nn import init

from lib.models.layers.Gate import SpatialGate, ChannelGate
from lib.utils.token_utils import token2patch, patch2token


class fusion_aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channels = dim

        self.linear = nn.Linear(dim * 2, self.channels)

        self.sg = SpatialGate()

        self.cg = ChannelGate(self.channels)

        # Initialize linear fusion layers with Kaiming initialization
        init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        # print("11111",x.shape) # ([2, 1536, 16, 16])

        x = x.flatten(2).transpose(1, 2).contiguous()
        # print("222222",x.shape) # ([2, 256, 1536])
        x_sum = self.linear(x)
        x_sum = token2patch(x_sum)
        x_sum = self.sg(x_sum) + self.cg(x_sum)
        x_fusion = x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print("333333",x_fusion.shape) # ([2, 768, 16, 16])

        return x1 * x_fusion, x2 * x_fusion

class GIM(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 2, act_layer=act_layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_v, x_i):
        B, C, H, W = x_v.shape
        N = int(H * W)

        x_v = patch2token(x_v)
        x_i = patch2token(x_i)
        # print("x_i = patch2token(x_i)",x_i.shape) ([2, 256, 768])

        x = torch.cat((x_v, x_i), dim=1)
        # print("x = torch.cat((x_v, x_i), dim=1)",x.shape) torch.Size([2, 512, 768])

        x = x + self.norm(self.mlp1(x))
        x_v, x_i = torch.split(x, (N, N,), dim=1)
        # print("x_v, x_i = torch.split(x, (N, N,), dim=1)",x_i.shape) ([2, 256, 768])

        x_v = token2patch(x_v)
        x_i = token2patch(x_i)
        # print("x_i = token2patch(x_i)",x_i.shape) ([2, 768, 16, 16])

        return x_v, x_i