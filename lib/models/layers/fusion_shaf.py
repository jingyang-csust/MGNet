import torch
from torch import nn

from torch.nn import init
from torch.nn.parameter import Parameter

from lib.models.layers.bi_dlk_adapt import Mlp, DropPath
from lib.utils.token_utils import token2patch, patch2token


class ShuffleAttention(nn.Module):

    def __init__(self, channel=768, G=12):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(channel * 2, channel)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # 扁平化
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x1):
        b, c, h, w = x1.size()
        # x = torch.cat((x1, x2), dim=1)
        # print("11111",x.shape) # ([2, 1536, 16, 16])

        # x = x.flatten(2).transpose(1, 2).contiguous()
        # print("222222",x.shape) # ([2, 256, 1536])
        # x_sum = self.linear(x)
        # x = token2patch(x_sum)
        # 将通道分成子特征
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # 通道分割
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # 通道注意力
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # 空间注意力
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # 沿通道轴拼接
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        out = self.channel_shuffle(out, 2)


        return x1 + out

class DLK(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.linear = nn.Linear(dim * 2,dim)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.MaxPool2d(1)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x1, x2):
        att1 = self.att_conv1(x1)
        att2 = self.att_conv1(x2)

        att = torch.cat([x1, x2], dim=1)  # ([2, 1536, 16, 16])
        att = token2patch(self.linear(patch2token(att)))  # ([2, 768, 16, 16])

        avg_att = self.avg(att) #  ([2, 768, 16, 16])
        avg_att = avg_att.expand_as(att)
        max_att = self.max(att) # ([2, 768, 16, 16])

        att = torch.cat([avg_att, max_att], dim=1)  # ([2, 1536, 16, 16])
        att = self.spatial_se(att)  # ([2, 768, 16, 16])
        # output = att1 * att[:, 0, :, :].unsqueeze(1) + att2 * att[:, 1, :, :].unsqueeze(1)
        output = att1 * att + att2 * att
        # output = output + x

        # output = output.permute(0, 2, 3, 1).reshape(B, N, C)  # 重塑形状
        return x1 + output, x2 + output

class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x1, x2):

        # print(x.shape)  [2, 768, 16, 16]
        shortcut1 = x1.clone()
        shortcut2 = x1.clone()
        x1 = self.proj_1(x1)
        x2 = self.proj_1(x2)
        x1 = self.act(x1)
        x2 = self.act(x2)
        x1, x2 = self.spatial_gating_unit(x1, x2)
        x1 = self.proj_2(x1)
        x2 = self.proj_2(x2)
        x1 = x1 + shortcut1
        x2 = x2 + shortcut2
        return x1, x2

class DLKBlock(nn.Module):
    def __init__(self, dim=768,drop_path=0.):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.channels = dim
        # self.linear = nn.Linear(dim * 2, self.channels)
        self.norm_layer = nn.LayerNorm(dim)
        self.norm_layer1 = nn.LayerNorm(dim)
        self.attn = DLKModule(dim)
        self.attn1 = DLKModule(dim)
        self.mlp = Mlp(dim)
        self.mlp1 = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x1, x2):
        x1_raw = x1
        x2_raw = x2

        x1 = self.norm_layer(patch2token(x1))
        x2 = self.norm_layer(patch2token(x2))

        x1, x2 = self.attn(token2patch(x1), token2patch(x2))

        # z_v = x_v[:,:lens_z,:]
        # x_v = x_v[:,lens_z:,:]
        # z_i = x_i[:,:lens_z,:]
        # x_i = x_i[:,lens_z:,:]
        #
        # z_v = token2patch(z_v)
        # x_v = token2patch(x_v)
        # z_i = token2patch(z_i)
        # x_i = token2patch(x_i)  # ([2, 768, 16, 16])
        #
        # # 融合rgb和rgbt的search
        # # x = torch.cat((x_v,x_i),dim=1)  # ([2, 1536, 16, 16])
        # # x = x.flatten(2).transpose(1, 2).contiguous()  #  ([2, 256, 1536])
        # # x = self.linear(x)  # ([2, 256, 768])
        # x = x_v + x_i
        # x = patch2token(x)
        #
        # shortcut = x.clone()
        # B, _, C = x.shape
        # x = x.transpose(1, 2).view(B, C, self.hx, self.wx)
        # x = (self.attn(x)).flatten(2).transpose(1, 2).contiguous()
        # x = shortcut + self.drop_path(self.layer_scale * x)
        # shortcut = x.clone()
        # x = ((self.norm_layer(x.transpose(1, 2).view(B, C, self.hx, self.wx)))
        #      .flatten(2).transpose(1, 2).contiguous())
        # x = self.mlp(x)
        # x = shortcut + self.drop_path(self.layer_scale * x)
        #
        # # 融合rgb和rgbt的search
        # # z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 8, 8])
        # # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 64, 1536])
        # # z = self.linear(z)  # ([2, 64, 768])
        # z = z_v + z_i  # ([2, 768, 8, 8])
        # z = patch2token(z)  # ([2, 64, 768])
        # shortcut = z.clone()
        # z = z.transpose(1, 2).view(B, C, self.hz, self.wz) # ([2, 768, 8, 8])
        # z = (self.attn(z)).flatten(2).transpose(1, 2).contiguous() # ([2, 64, 768])
        # z = shortcut + self.drop_path(self.layer_scale * z) # ([2, 64, 768])
        # shortcut = z.clone()
        # z = ((self.norm_layer(z.transpose(1, 2).view(B, C, self.hz, self.wz)))
        #      .flatten(2).transpose(1,2).contiguous())  # ([2, 64, 768])
        # z = self.mlp(z)
        # z = shortcut + self.drop_path(self.layer_scale * z)
        #
        # xo = torch.cat((z,x),dim=1)
        return x1 + x1_raw, x2 + x2_raw
# x = torch.ones(2,768,16,16)
# m = DLK(768)
# o_i,o_v = m(x,x)
# print(o_v.shape)