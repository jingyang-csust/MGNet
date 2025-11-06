import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, in_channel,out_channel, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            ChannelAttention(out_channel, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

# x = torch.ones(2,768 * 2,16,16)
# m = ChannelAttentionBlock(768 * 2,768)
# o_v = m(x)
# print(o_v.shape)