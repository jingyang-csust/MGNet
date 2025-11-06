import torch
from torch import nn

# from lib.utils.token_utils import patch2token, token2patch
#
#
# class SMFFL(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=2, hidden_dim=None):
#         super(SMFFL, self).__init__()
#         self.in_channels = in_channels
#         self.reduction_ratio = reduction_ratio
#         self.hidden_dim = hidden_dim if hidden_dim else in_channels // reduction_ratio
#
#         # 使用线性层减少维度
#         self.linear1 = nn.Linear(in_channels, self.hidden_dim)
#         self.linear2 = nn.Linear(in_channels, self.hidden_dim)
#
#         # 使用不同核大小的深度卷积层
#         self.dw_conv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)
#         self.dw_conv5x5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, groups=self.hidden_dim)
#
#         # 还原维度的线性层
#         self.gelu = nn.GELU()
#         self.linear3 = nn.Linear(self.hidden_dim * 2, in_channels)
#
#         # 输入的LayerNorm
#         self.ln = nn.LayerNorm(in_channels)
#
#     def forward(self, x):
#         x = patch2token(x)
#         # 应用LayerNorm
#         x_ln = self.ln(x)
#
#         # 第一个分支：3x3深度卷积
#         x1 = self.linear1(x_ln)
#         x1 = token2patch(x1)
#         # x1 = x1.permute(0, 3, 1, 2)  # 转换为(B, C, H, W)以适应卷积操作
#         x1 = self.dw_conv3x3(x1)
#         # x1 = x1.permute(0, 2, 3, 1)  # 转回(B, H, W, C)
#
#         # 第二个分支：5x5深度卷积
#         # x_ln = patch2token(x_ln)
#         x2 = self.linear2(x_ln)
#         # x2 = x2.permute(0, 3, 1, 2)  # 转换为(B, C, H, W)以适应卷积操作x1
#         x2 = token2patch(x2)
#         x2 = self.dw_conv5x5(x2)
#         # x2 = x2.permute(0, 2, 3, 1)  # 转回(B, H, W, C)
#
#         # 将两个分支的特征连接起来
#         x_concat = torch.cat([x1, x2], dim=1)
#
#         # 应用GELU和最终的线性变换
#         x_out = self.gelu(x_concat)
#         x_out = patch2token(x_out)
#         x_out = self.linear3(x_out)
#
#         # 残差连接
#         out = x + x_out
#
#         out = token2patch(out)
#
#         return out

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        #self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.conv = nn.Conv2d(
            dim * 2,
            dim * 2,
            3,
            stride=1,
            padding=1
        )

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_atten = nn.Sequential(
        #     nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv(self.avgpool(output) + self.maxpool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)

        output = output * att
        return output

x = torch.ones(2,768,16,16)
m = DFF(768)
o_v = m(x, x)
print(o_v.shape)