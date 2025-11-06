import numpy as np
import torch
# from timm.models import Mlp
from torch import nn
from torch.nn import init, Parameter
from einops import rearrange

from timm.models.layers import Mlp

from lib.models.layers.cwt import DecoderCFALayer
from lib.utils.token_utils import token2patch, patch2token
# from model.encoders.vmamba import Bi_direct_adapter
import torch.nn.functional as F

from model.encoders.vmamba import VSSBlock


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Attention_Module(nn.Module):
    def __init__(self, dim, reduction=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.hidden_dim = dim // reduction
        self.linear = nn.Linear(dim, self.hidden_dim * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = MutualAttention(self.hidden_dim, num_heads=8)
        self.end_proj = nn.Linear(self.hidden_dim, dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x1, x2):
        x1 = patch2token(x1)
        x2 = patch2token(x2)
        y1, u1 = self.act1(self.linear(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.linear(x2)).chunk(2, dim=-1)
        v1 = self.cross_attn(u1, u2)

        y1 = y1 + v1

        out_x1 = self.norm1(x1 + self.end_proj(y1))
        out_x1 = token2patch(out_x1)
        return out_x1

class Bi_direct_adapter(nn.Module):
    def __init__(self,in_dim, hidden_dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(in_dim, hidden_dim)
        self.adapter_up = nn.Linear(hidden_dim, in_dim)
        self.adapter_mid = nn.Linear(hidden_dim,hidden_dim)
        # self.adapter_mid = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        # self.dropout = nn.Dropout(0.1)
        # self.dim = hidden_dim

    def forward(self, x):
        x = patch2token(x)
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return token2patch(x_up)


class Bi_direct_adapter1(nn.Module):
    def __init__(self,in_dim, hidden_dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(in_dim, hidden_dim)
        self.adapter_up = nn.Linear(hidden_dim, in_dim)
        self.adapter_mid = nn.Linear(hidden_dim,hidden_dim)
        # self.adapter_mid = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        # self.dropout = nn.Dropout(0.1)
        # self.dim = hidden_dim

    def forward(self, x):
        # x = patch2token(x)
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        # return token2patch(x_up)
        return x_up

class GAM_Attention(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class SE(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # B, _, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape) #  ([2, 768, 16, 16])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape) ([2, 768, 1, 1])
        x_out =x * y.expand_as(x)
        return x_out

class CBAMLayer(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class CB11(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2).contiguous()

class DWC(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()

class LSA(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, c1,c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CB11(c2)
        # self.pwconv2 = CB11(c1)
        self.dwconv3 = DWC(c2, 3)
        self.dwconv5 = DWC(c2, 5)
        self.dwconv7 = DWC(c2, 7)
        # self.pwconv2 = CB11(c2)
        self.fc2 = nn.Linear(c2, c1)
        # self.eca = ECAAttention()
        # Initialize fc1 layer with Kaiming initialization
        # init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x) :
        B, C, H, W = x.shape
        x = patch2token(x)
        # print(x.shape)
        # raw = x
        x = self.fc1(x)
        # raw = x
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        # xo = self.fc2(patch2token(sxelf.eca(token2patch(raw + x + x1 + x2))))
        xo = self.fc2(x + x1 + x2 + x3)
        xo = token2patch(xo)
        return xo

class Attention(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # self.fuse = ECAAttention()
        #
        self.norm1 = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, dim // 2)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(dim // 2, dim)
        self.hidden_dim = dim
        self.num_heads = num_heads
        head_dim = self.hidden_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.norm2 = nn.LayerNorm(dim)
        self.ro = Router(dim // 2)

    def forward(self, x):
        x = patch2token(x)
        res = x
        x = self.norm1(x)
        # x = self.act(self.down(x))
        # res1 = x
        # x = patch2token(self.ro(token2patch(x)))

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # x = res1 + x
        # x = res + self.up(x)
        x = res + x
        # x = self.norm2(x)
        # x = self.norm(x)
        x = token2patch(x)
        # x = self.fuse(x)
        return x

class Attention11(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # self.fuse = ECAAttention()
        self.hidden_dim = dim
        self.num_heads = num_heads
        head_dim = self.hidden_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x = patch2token(self.ro(token2patch(x)))
        # x = patch2token(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.norm(x)
        # x = token2patch(x)
        # x = self.fuse(x)
        return x

class Attention2(nn.Module):
    """
    输入：B N C
    输出：B N C
    """
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=8):
        super(Attention2, self).__init__()
        self.down = nn.Linear(in_dim1, in_dim1 // 2)
        self.fuse = ECAAttention()

        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(in_dim1 // 2, in_dim1)
        self.hidden_dim = in_dim1 // 2
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.norm = nn.LayerNorm(in_dim1)

        self.proj_q1 = nn.Linear(self.hidden_dim, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(self.hidden_dim, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(self.hidden_dim, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, self.hidden_dim)

    def forward(self, x, y, mask=None):

        x = patch2token(x)
        y = patch2token(y)
        res = x

        batch_size, seq_len1, in_dim1 = y.size()
        seq_len2 = y.size(1)
        x = self.act(self.down(x))
        y = self.act(self.down(y))
        res1 = x

        q1 = self.proj_q1(x).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(y).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(y).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        output = res1 + output

        output = token2patch(self.norm(res + self.up(output)))

        return output


class CrossAttention(nn.Module):
    """
    输入：B N C
    输出：B N C
    """
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.down = nn.Linear(in_dim1, in_dim1 // 2)
        self.fuse = ECAAttention()

        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(in_dim1 // 2, in_dim1)
        self.hidden_dim = in_dim1 // 2
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(self.hidden_dim, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(self.hidden_dim, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(self.hidden_dim, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, self.hidden_dim)

    def forward(self, x, y, z, mask=None):

        x = patch2token(x)
        y = patch2token(y)
        z = patch2token(z)
        res = x

        batch_size, seq_len1, in_dim1 = y.size()
        seq_len2 = y.size(1)
        x = self.act(self.down(x))
        y = self.act(self.down(y))
        z = self.act(self.down(z))

        q1 = self.proj_q1(x).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(y).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(y).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        output = token2patch(res + self.up(output))

        return output

class MutualAttention(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super(MutualAttention, self).__init__()
        self.down = nn.Linear(dim, dim // 2)
        # self.fuse = ECAAttention()
        # self.fusion = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        #
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(dim // 2, dim)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.hidden = dim // 2
        # self.adapter1 = Bi_direct_adapter(self.hidden)
        # self.adapter2 = Bi_direct_adapter(self.hidden)
        # self.act = nn.Sigmoid()
        # self.fusion = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.lsa = LSA(dim//2,dim//2)
        # self.norm1 = nn.LayerNorm(dim)
        # self.linear = nn.Linear(self.hidden*2,self.hidden)
        # self.linear2 = nn.Linear(self.hidden*2,self.hidden)
        self.q = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        # self.q1 = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        z_v = x[:,:64,:]
        x_v = x[:,64:,:]

        z_i = y[:,:64,:]
        x_i = y[:,64:,:]

        # x = token2patch(x)
        # y = token2patch(y)
        res = z_v
        x = self.act(self.down(z_v))
        y = self.act(self.down(z_i))
        x = token2patch(x)
        y = token2patch(y)
        res1 = x
        b, c, h, w = x.shape
        q = self.q(y)
        k = self.k(x) # imag0-=po
        v = self.v(x) # image

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        out = res1 + out
        # out  = self.lsa(out)
        out = res + self.up(patch2token(out))


        res2 = x_v
        x1 = self.act(self.down(x_v))
        y1 = self.act(self.down(x_i))
        x1 = token2patch(x1)
        y1 = token2patch(y1)
        res3 = x1
        b, c, h, w = x1.shape
        q1 = self.q(y1)
        k1 = self.k(x1) # imag0-=po
        v1 = self.v(x1) # image

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out1 = self.project_out(out1)

        out1 = res3 + out1
        # out  = self.lsa(out)
        out1 = res2 + self.up(patch2token(out1))


        xo = torch.cat((out, out1), dim=1)
        # out = self.norm1(patch2token(out))
        # out = token2patch(out)
        # out = self.fuse(out)
        # out = out * self.global_att(out)
        return xo

class MutualAttentionNo(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super(MutualAttentionNo, self).__init__()
        # self.down = nn.Linear(dim, dim // 2)
        # self.fuse = ECAAttention()
        # self.fusion = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        #
        # self.act = nn.ReLU(inplace=True)
        # self.up = nn.Linear(dim // 2, dim)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.hidden = dim
        # self.adapter1 = Bi_direct_adapter(self.hidden)
        # self.adapter2 = Bi_direct_adapter(self.hidden)
        # self.act = nn.Sigmoid()
        # self.fusion = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.lsa = LSA(dim//2,dim//2)
        # self.norm1 = nn.LayerNorm(dim)
        # self.linear = nn.Linear(self.hidden*2,self.hidden)
        # self.linear2 = nn.Linear(self.hidden*2,self.hidden)
        self.q = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        # self.q1 = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(self.hidden, self.hidden, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        res = x
        # x = patch2token(x)
        # y = patch2token(y)
        # z = patch2token(z)
        # x = torch.cat((x,y),dim=2)
        # x1 = self.fusion(x)
        # x = self.act(self.down(x))
        # y = self.act(self.down(y))
        # z = self.act(self.down(z))
        # x = token2patch(x)
        # y = token2patch(y)
        # z = token2patch(z)
        # res1 = x
        # x1 = self.lsa(x1)
        # res_x = x
        # res_y = y

        b, c, h, w = x.shape

        # q_res = self.adapter1(x)
        # k_res = self.adapter2(y)
        q = self.q(y)
        # q = torch.cat((self.q(x),self.q(y)),dim=1) # image
        # q = token2patch(self.linear(patch2token(q))) # image
        k = self.k(x) # imag0-=po
        v = self.v(x) # image
        # k = torch.cat((self.k(x),self.k(y)),dim=1)  # event
        # k = token2patch(self.linear(patch2token(k)))
        # v = torch.cat((self.v(x),self.v(y)),dim=1)  # event
        # v = token2patch(self.linear(patch2token(v)))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        # out = res1 + out
        # out  = self.lsa(out)
        out = res + out
        # out = self.norm1(patch2token(out))
        # out = token2patch(out)
        # out = self.fuse(out)
        # out = out * self.global_att(out)
        # out = patch2token(out)
        return out

class MixAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x):
        """
        x is a concatenated vector of template and search region features.
        """
        t_h = 8
        t_w = 8
        s_h = 16
        s_w = 16
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ShuffleAttention1(nn.Module):

    def __init__(self, channel=8, G=1):
        super().__init__()
        self.linear = nn.Linear(channel * 2,channel)
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, x1,x2):
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        b, c, h, w = x.size()
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
        return x1 + out,x2 + out

class Mlp2(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = patch2token(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = token2patch(x)
        return x

class MA_SE2(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, channel, reduction=16):
        super(MA_SE2, self).__init__()
        self.linear = nn.Linear(channel * 2,channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x = token2patch(x)
        # B, _, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape) #  ([2, 768, 16, 16])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape) ([2, 768, 1, 1])
        x_out =x * y.expand_as(x)
        return x_out

class LPU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.conv(x) + x

class mamba_fusion5(nn.Module):
    def __init__(self,
                 dim,
                 dt_rank="auto",
                 d_state=4,
                 ssm_ratio=2.0,
                 attn_drop_rate=0.,
                 drop_rate=0.0,
                 mlp_ratio=4.0,
                 drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 shared_ssm=False,
                 softmax_version=False,
                 use_checkpoint=False,
                 dims=768,
                 ape=False,
                 drop_path_rate=0.2,):
        super().__init__()

        # self.ape = ape

        # self.fusion = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )

        self.vss = VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
        # self.eca = ECAAttention(dim)
    def forward_features(self, x1):
        """
        input:  B x N x C
        output: B x N x C
        """
        x_fuse = self.vss(x1)
        return x_fuse

    def forward(self, x1):
        # x2 = patch2token(x2)
        out = self.forward_features(x1)
        # print(out.shape)
        # out = self.eca(out)
        return out

class SMFFL(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, hidden_dim=None):
        super(SMFFL, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = hidden_dim if hidden_dim else in_channels // reduction_ratio

        # 使用线性层减少维度
        self.linear1 = nn.Linear(in_channels, self.hidden_dim)
        self.linear2 = nn.Linear(in_channels, self.hidden_dim)

        # 使用不同核大小的深度卷积层
        self.dw_conv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)
        self.dw_conv5x5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, groups=self.hidden_dim)

        # 还原维度的线性层
        self.gelu = nn.GELU()
        self.linear3 = nn.Linear(self.hidden_dim * 2, in_channels)

        # 输入的LayerNorm
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        x = patch2token(x)
        # 应用LayerNorm
        x_ln = self.ln(x)

        # 第一个分支：3x3深度卷积
        x1 = self.linear1(x_ln)
        x1 = token2patch(x1)
        # x1 = x1.permute(0, 3, 1, 2)  # 转换为(B, C, H, W)以适应卷积操作
        x1 = self.dw_conv3x3(x1)
        # x1 = x1.permute(0, 2, 3, 1)  # 转回(B, H, W, C)

        # 第二个分支：5x5深度卷积
        # x_ln = patch2token(x_ln)
        x2 = self.linear2(x_ln)
        # x2 = x2.permute(0, 3, 1, 2)  # 转换为(B, C, H, W)以适应卷积操作x1
        x2 = token2patch(x2)
        x2 = self.dw_conv5x5(x2)
        # x2 = x2.permute(0, 2, 3, 1)  # 转回(B, H, W, C)

        # 将两个分支的特征连接起来
        x_concat = torch.cat([x1, x2], dim=1)

        # 应用GELU和最终的线性变换
        x_out = self.gelu(x_concat)
        x_out = patch2token(x_out)
        x_out = self.linear3(x_out)

        # 残差连接
        out = x + x_out

        out = token2patch(out)

        return out

class AFF(nn.Module):
    '''
        输入: B C H W
        输出：B C H W
    '''

    def __init__(self, in_features,
                 r=4):
        super(AFF, self).__init__()
        # self.down = nn.Linear(in_features,in_features // 2)
        # self.up = nn.Linear(in_features // 2,in_features)
        self.linear = nn.Linear(in_features * 2, in_features)
        self.linear2 = nn.Linear(in_features * 2, in_features)
        inter_channels = int(in_features // r)
        self.channels = in_features
        # self.se = MA_SE2(self.channels)
        self.local_att = nn.Sequential(
            nn.Conv2d(self.channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
        )
        # self.mamba = mamba_fusion5(self.channels)
        # self.sum = SMFFL(self.channels)

        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.channels),
        # )

        # self.global_att2 = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.channels),
        # )

        # self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2, act_layer=nn.GELU)
        # self.norm3 = nn.LayerNorm(self.channels)
        # self.norm2 = nn.LayerNorm(self.channels)
        # self.fc = nn.Sequential(
        #     nn.Conv2d(self.channels, self.channels * 2, kernel_size=1, stride=1,
        #                          padding=0),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(self.channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, stride=1,
        #                            padding=0),
        #     nn.BatchNorm2d(self.channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        # )
        self.att = MS_FFN(self.channels)
        # self.ms = MultiScaleDWConv(self.channels)
        self.eca = ECAAttention()
        # self.sg = SpatialGroupEnhance()
        # self.attn1 = ChannelAttention(self.channels)
        # self.attn2 = ChannelAttention(self.channels)
        # self.ro = Router(self.channels)
        # self.ro2 = Router(self.channels)
        # self.ro3 = Router(self.channels)
        # self.adapter1 = Bi_direct_adapter(self.channels)
        # self.adapter2 = Bi_direct_adapter(self.channels)
        self.mlp = Mlp2(self.channels)
        self.norm = nn.BatchNorm2d(self.channels)
        # self.mlp3 = Mlp2(self.channels)
        # self.cfn = CFN(self.channels, self.channels)
        # self.vh = DecoderCFALayer(self.channels)

        # self.sigmoid = nn.Sigmoid()
        # self.act = nn.Sigmoid()


    def forward(self, x1, x2):
        # x1 = token2patch(self.down(patch2token(x1)))
        # x2 = token2patch(self.down(patch2token(x2)))
        # res_x1 = x1
        # x1 = self.ro(x1)
        # x1 = x1 + token2patch(self.norm(self.mlp2(x1)))

        # res_x2 = x2
        # x2 = self.ro2(x2)
        # x2 = x2 + token2patch(self.norm2(self.mlp3(x2)))

        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        # x_flag = self.sum(x)
        x_flag = x
        # xl = self.local_att(x)
        # x = x1 + x2
        xg = self.local_att(x_flag)
        x_s = self.eca(x_flag)
        # x_p = self.mamba(x)
        # xlg = xg + x_s + x_p
        xlg = xg + x_s
        # wei = self.eca(xlg)
        # wei = self.sigmoid(xlg)
        # out = x * wei
        x1_out = x1 + xlg
        x2_out = x2 + xlg
        # out = x1_out + x2_out
        # out = self.sum(out)
        # print(out.shape)
        # out = self.fc(out)
        # print(out.shape)
        # out = 2 * x1 * wei + 2 * x2 * (1 - wei)
        x = torch.cat((x1_out, x2_out), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        out = self.linear2(x)
        out = token2patch(out)
        # out = self.fc(x)
        # x = patch2token(out)
        # out = out + self.norm3(self.mlp1(out))
        # out = token2patch(out)

        # out = out + self.fc(out)
        # out = self.norm(out)
        # B, C, H, W = out.shape
        # out = self.cfn(out, H, W)
        out = out + self.norm(self.mlp(out))

        # out = self.mamba(out)
        # out = self.se(out)
        # out = token2patch(self.up(patch2token(out)))
        # x2 = token2patch(self.up(patch2token(x2)))


        return out
class AFF2(nn.Module):
    '''
        输入: B C H W
        输出：B C H W
    '''

    def __init__(self, in_features,
                 h,
                 w,
                 hidden_features=None,
                 out_features=None,
                 r=4):
        super(AFF2, self).__init__()
        self.linear = nn.Linear(in_features * 2,in_features)
        self.linear2 = nn.Linear(in_features * 2,in_features)
        inter_channels = int(in_features // r)
        self.channels = in_features
        self.local_att = nn.Sequential(
            nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        # self.global_att2 = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.channels),
        # )

        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2, act_layer=nn.GELU)
        self.norm = nn.LayerNorm(self.channels)
        self.att = MS_FFN(self.channels)
        self.ms = MultiScaleDWConv(self.channels)
        self.eca = ECAAttention()
        self.fc = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=1, stride=1,
                                 padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, stride=1,
                                   padding=0),
            nn.BatchNorm2d(self.channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.norm = nn.BatchNorm2d(self.channels)
        self.sg = SpatialGroupEnhance()
        self.attn = ChannelAttention(self.channels)
        self.ro = Router(self.channels)
        self.adapter1 = Bi_direct_adapter(self.channels)
        self.adapter2 = Bi_direct_adapter(self.channels)
        #
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1 = self.adapter1(x1)
        # x2 = self.adapter1(x2)
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        # xl = self.local_att(x)
        # x = x1 + x2
        xg = self.att(x)
        x_s = self.eca(x)
        xlg = xg + x_s
        # wei = self.eca(xlg)
        wei = self.sigmoid(xlg)
        # out = x * wei
        x1_out = x1 * wei
        x2_out = x2 * (1 - wei)
        # out = x1_out + x2_out
        # print(out.shape)
        # out = self.fc(out)
        # print(out.shape)
        # out = 2 * x1 * wei + 2 * x2 * (1 - wei)
        # x = torch.cat((x1_out, x2_out), dim=1)
        # out = self.fc(x)
        # x = patch2token(out)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # out = self.linear2(x)
        # out = x + self.norm(self.mlp1(x))
        # out = token2patch(out)

        # out = out + self.fc(out)
        # out = self.norm(out)
        # out = x +  self.ro(out)


        return x1_out, x2_out

class SpatialGroupEnhance(nn.Module):
    """
        input:b c h w
        output:b c h w
    """
    def __init__(self, groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


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

    def forward(self, x):

        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w
        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=2):
        super(ChannelAttention, self).__init__()
        # self.linear = nn.Linear(num_feat * 2,num_feat)
        # self.eca = ECAAttention()
        # self.se = SE(num_feat)
        # self.se2 = SE(num_feat)
        # self.adapter1 = Bi_direct_adapter(num_feat)
        # self.adapter2 = Bi_direct_adapter(num_feat)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 3,stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 3,stride=1, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # self.adapt = Bi_direct_adapter(num_feat)

    def forward(self, x):
        # res = self.adapt(x)
        # res = x
        # x1 = self.adapter1(x1)
        # x2 = self.adapter1(x2)
        # x = torch.cat((x1, x2), dim=1)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # x_sum = self.linear(x)
        # x = token2patch(x_sum)
        # out1 = self.se(x)
        # x = token2patch(x)
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc1(self.max_pool(x))
        attn = avg_out + max_out
        out = x * self.sigmoid(attn)
        # out = out2 + out1
        # x1_out = x1 + out
        # x2_out = x2 + out
        # return x1_out, x2_out
        # out = patch2token(out)
        return out


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=2, squeeze_factor=2):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1)
            )

        # self.res = MultiScaleDWConv(num_feat)
        self.sum = ChannelAttention(num_feat, squeeze_factor)

    def forward(self, x):
        z = x[:,:64,:]
        x = x[:,64:,:]
        z = token2patch(z)
        x = token2patch(x)

        x = self.cab(x)
        z = self.cab(z)
        # x = self.res(x)

        xo = self.sum(x)
        zo = self.sum(z)
        xo = patch2token(xo)
        zo = patch2token(zo)
        out = torch.cat((zo,xo),dim=1)

        return out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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

    def forward(self, x):

        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        x = x * y.expand_as(x)

        return x

class ECAAttention2(nn.Module):

    def __init__(self,dim, kernel_size=3):
        super().__init__()
        self.linear = nn.Linear(dim * 2,dim)
        self.linear2 = nn.Linear(dim * 2,dim)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
        self.adapt = Bi_direct_adapter(dim)

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

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        res_x1 = self.adapt(x1)
        res_x2 = self.adapt(x2)

        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        x1 = res_x1 + x * y.expand_as(x)
        x2 = res_x2 + x * y.expand_as(x)

        # x = torch.cat((x1, x2), dim=1)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # out = self.linear2(x)
        # out = token2patch(out)

        return x1, x2

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

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class MS_FFN(nn.Module):  ### MS-FFN
    """
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0, ):
        super().__init__()
        hidden_features = in_features // 2
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_features),
        )
        self.drop = nn.Dropout(drop)
        # self.adapter1 = Bi_direct_adapter(hidden_features)
        # self.ro = Router(hidden_features)
        # self.eca = ECAAttention(hidden_features)
        # self.ldc = LDC(hidden_features, hidden_features)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(dim * 2, dim * 2, 1, stride=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(dim * 2),
        #     nn.Conv2d(dim * 2, dim, 1, stride=1, padding=0),
        #     nn.BatchNorm2d(dim)
        # )
        # self.attn = Dff_fusion(dim * 2)
        # self.mamba1 = mamba_fusion5(dim * 2)
        # self.mamba2 = mamba_fusion5(dim)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_atten = nn.Sequential(
        #     nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.conv = nn.Conv2d(
            dim * 2,
            dim * 2,
            3,
            stride=1,
            padding=1
        )
        self.nonlin = nn.Sigmoid()
        # self.adapter1 = Bi_direct_adapter(dim)
        self.att = MultiScaleDWConv(dim * 2)
        # self.mlp = Mlp2(dim)
        # self.norm = nn.BatchNorm2d(dim)
        # self.ldc = LDC(dim * 2, dim)

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.att(self.avgpool(output) + self.maxpool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)

        output = output * att
        return output

class Router(nn.Module):
    def __init__(self, embed_size):
        super(Router, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.maxpool1 = nn.AdaptiveMaxPool2d(1)
        #self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        # self.conv = nn.Conv2d(
        #     embed_size * 2,
        #     embed_size,
        #     3,
        #     stride=1,
        #     padding=1
        # )
        self.ms = MS_FFN(in_features=embed_size * 2,out_features=embed_size)
        self.ms1 = MS_FFN(in_features=embed_size * 2,out_features=embed_size)
        self.act = nn.Sigmoid()
        # self.adapt = Bi_direct_adapter(embed_size)


    def forward(self, x):
        z = x[:,:64,:]
        x = x[:,64:,:]
        z = token2patch(z)
        x = token2patch(x)

        raw = x
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out,avg_out],1)
        x = self.ms(f_out)
        x = self.act(x)
        x = x * raw

        raw_z = z
        avg_z_out = self.avgpool1(z)
        max_z_out = self.maxpool1(z)
        f_z_out = torch.cat([max_z_out, avg_z_out], 1)
        z = self.ms1(f_z_out)
        z = self.act(z)
        z = z * raw_z

        xo = patch2token(x)
        zo = patch2token(z)
        out = torch.cat((zo,xo), dim=1)

        return out

class Dff_fusion(nn.Module):
    def __init__(self, embed_size):
        super(Dff_fusion, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        #self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.conv = nn.Conv2d(
            embed_size * 2,
            embed_size,
            3,
            stride=1,
            padding=1
        )
        self.act = nn.Sigmoid()
        # self.adapt = Bi_direct_adapter(embed_size)


    def forward(self, x):
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out,avg_out],1)
        x = self.conv(f_out)
        x = self.act(x)
        return x

class Router2(nn.Module):
    def __init__(self, embed_size):
        super(Router2, self).__init__()
        self.linear = nn.Linear(embed_size * 2,embed_size)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        #self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.conv = nn.Conv2d(
            embed_size * 2,
            embed_size,
            3,
            stride=1,
            padding=1
        )
        self.act = nn.Sigmoid()


    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        raw = x
        #x = x.reshape(x.shape[0],-1)
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out,avg_out],1)

        # x = f_out.contiguous().view(f_out.size(0), -1)
        # print(x.shape)
        x = self.conv(f_out)
        x= self.act(x)
        # x = x * raw
        x1_out = x1 + x * x2
        x2_out = x2 + x * x1
        return x1_out, x2_out

class Mix_Frequency(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super(Mix_Frequency,self).__init__()
        self.down = nn.Linear(dim,dim // 2)
        self.up = nn.Linear(dim // 2,dim)
        self.linear = nn.Linear(dim,dim // 2)
        self.mlp1 = Mlp(in_features=dim // 2, hidden_features=dim, act_layer=act_layer)
        self.norm = nn.LayerNorm(dim // 2)
        self.norm1 = nn.LayerNorm(dim // 2)
        self.norm2 = nn.LayerNorm(dim // 2)
        # Initialize complex weights
        # self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 1x1 convolution and GELU activation
        self.conv1x1_real = nn.Linear(dim // 2, dim // 2)
        self.conv1x1_imag = nn.Linear(dim // 2, dim // 2)
        self.act = nn.ReLU()

    def forward(self, x, y, rate = 0.5, dimx = 1):

        x = token2patch(self.down(patch2token(x)))
        y = token2patch(self.down(patch2token(y)))
        xy = torch.cat((x, y), dim=1)
        xy = xy.flatten(2).transpose(1, 2).contiguous()
        xy = self.linear(xy) # ([2, 256, 384])

        # xy = torch.cat([x,y],dim=dimx) # ([2, 512, 768])

        xy_f = torch.fft.rfft(xy,dim=dimx)

        xy_f_local = xy_f # ([2, 129, 384])
        x_fft_real = xy_f_local.real # ([2, 129, 384])
        x_fft_real = self.norm1(self.conv1x1_real(x_fft_real)) # ([2, 129, 384])
        x_fft_imag = xy_f_local.imag
        x_fft_imag = self.norm2(self.conv1x1_imag(x_fft_imag))
        x_fft = torch.complex(x_fft_real, x_fft_imag)
        x_fft = torch.fft.irfft(x_fft,dim=dimx)
        # print(x_fft.shape) # ([2, 256, 384])


        # xy_f = torch.fft.rfft2(xy, dim=(1, 2), norm='ortho')
        # print(xy_f.shape) ([2, 257, 768])

        """
        生成一个与 xy_f 形状相同的随机掩码 m，其中每个元素根据 rate 的概率被设置为 True。
        接着，计算 xy_f 的振幅 amp，并按振幅大小排序，取出最主要的频率分量（掩码为 dominant_mask），与 m 进行与操作。
        根据掩码 m 将 xy_f 的实部和虚部中对应的部分填充为零。
        """
        m = torch.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=dimx, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        # print(freal.shape) ([2, 257, 768])
        # print(xy_f.shape) ([2, 257, 768])

        """
        生成一个随机打乱的索引 b_idx，用它来随机重排 x 和 y，得到新的 x2 和 y2。
        然后，将 x2 和 y2 进行拼接，并再次进行快速傅里叶变换，得到 xy2_f。
        """
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]

        xy2 = torch.cat([x2,y2],dim=dimx)
        xy2 = xy2.flatten(2).transpose(1, 2).contiguous()
        xy2 = self.linear(xy2)

        xy2_f = torch.fft.rfft(xy2,dim=dimx)
        # print(xy2_f.shape) ([2, 257, 768])

        """
        对掩码 m 取反，并用这个反掩码将 xy2_f 的实部和虚部对应部分填充为零，得到 freal2 和 fimag2。
        """
        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        """
        将之前填充后的 freal 和 fimag 与 freal2 和 fimag2 相加，混合它们的频率分量。
        """
        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)

        xy = torch.fft.irfft(xy_f,dim=dimx) + x_fft

        # xy = xy + self.norm(self.mlp1(xy))
        xy = token2patch(self.up(xy))

        return xy
class ShuffleAttention(nn.Module):

    def __init__(self, channel=8, G=1):
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

    def forward(self, x):
        b, c, h, w = x.size()
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
        return out


class SHA_Fusion(nn.Module):
    """
    BNC
    BNC
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim

        self.sha = ShuffleAttention(dim)
        # self.SA = Attention(dim)

        # self.sha_z = ShuffleAttention(dim)

    def forward(self,x):
        # z_v = x[:, :64, :]
        # x_v = x[:, 64:, :]

        x = token2patch(x)
        # x_v = token2patch(x_v)

        x = self.sha(x)  # ([2, 768, 16, 16])
        x = patch2token(x)

        # z = self.sha_z(z_v)  # ([2, 768, 16, 16])
        # z = patch2token(z)

        # x = torch.cat((z,x),dim=1)
        return x

class Attention_Module1(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, 8)
        # self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = SHA_Fusion(8)
        # self.cross_attn = Router(8)
        # self.cross_attn = ChannelAttention(8)
        self.end_proj = nn.Linear(8, dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x1):
        # y1, u1 = self.act1(self.linear(x1))
        # y2, u2 = self.act2(self.linear(x2))
        # v1, v2 = self.cross_attn(u1, u2)
        res = patch2token(x1)
        y = self.linear(res)
        v = self.cross_attn(y)

        y1 = y + v

        out_x1 = self.norm1(res + self.end_proj(y1))
        out_x1 = token2patch(out_x1)
        return out_x1

class CFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        #
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                                groups=in_channels)
        self.bn33 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                                groups=in_channels)
        self.bn11 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        #
        self.conv_up = nn.Linear(in_channels, in_channels * 2)
        self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Linear(in_channels * 2, in_channels)
        self.bn_down = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # down
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

        # norm all
        self.norm = nn.BatchNorm2d(out_channels)
        self.adapter = Bi_direct_adapter(out_channels)


    def forward(self, x, H, W):
        # x = patch2token(x)
        # B, N, _C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        # print(f"x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous():{x.shape}") ([2, 768, 16, 16])
        residual = self.residual(x)

        #  + skip-connection
        x = x + self.bn11(self.conv11(x)) + self.bn33(self.conv33(x))

        #  + skip-connection
        # x = x + self.bn_down(token2patch(self.conv_down(patch2token(self.act(self.bn_up(token2patch(self.conv_up(patch2token(x)))))))))
        x = self.adapter(x)

        x = self.adjust(x)

        out = self.norm(residual + x)
        return out

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    """
    输入：B C H W
    输出：B C H E
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class LDC(nn.Module):
    """
    input : B C H W
    output : B C H W
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]
        # print(self.learnable_mask[:, :, None, None].shape)

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        z = x[:,:64,:]
        x = x[:,64:,:]
        z = token2patch(z)
        x = token2patch(x)

        outx_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)


        outz_diff = F.conv2d(input=z, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)

        xo = patch2token(outx_diff)
        zo = patch2token(outz_diff)
        out = torch.cat((zo,xo), dim=1)

        return out