import torch
from torch import nn
from torch.nn import init

from lib.models.layers.MA import MutualAttention, ChannelAttentionBlock, mamba_fusion5, LPU, Router, MS_FFN, LDC
from lib.models.layers.shaf import Mlp
from lib.utils.token_utils import token2patch, patch2token


class Attention(nn.Module):
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


    def forward(self, x1, x2):
        """
        x is a concatenated vector of template and search region features.
        """

        z = x1[:,:64,:]
        x = x2[:,64:,:]

        x = torch.cat((z,x), dim=1)

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

class Attention2(nn.Module):
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

class Attention_Module(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.hidden = 8
        self.linear = nn.Linear(dim, self.hidden)
        self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = Attention(self.hidden, num_heads=8)
        self.end_proj = nn.Linear(self.hidden, dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x1, x2):
        # y1, u1 = self.act1(self.linear(x1))
        # y2, u2 = self.act2(self.linear(x2))
        # v1, v2 = self.cross_attn(u1, u2)
        y1 = self.act1(self.linear(x1))
        y2 = self.act1(self.linear(x2))
        v1 = self.cross_attn(y1, y2)

        y1 = y1 + v1

        out_x1 = self.norm1(x1 + self.end_proj(y1))
        return out_x1

class Attention_Module2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.hidden = 8
        self.linear = nn.Linear(dim, self.hidden)
        self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = Attention2(self.hidden, num_heads=8)
        self.end_proj = nn.Linear(self.hidden, dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x1):
        # y1, u1 = self.act1(self.linear(x1))
        # y2, u2 = self.act2(self.linear(x2))
        # v1, v2 = self.cross_attn(u1, u2)
        y1 = self.act1(self.linear(x1))
        v1 = self.cross_attn(y1)

        y1 = y1 + v1

        out_x1 = self.norm1(x1 + self.end_proj(y1))
        return out_x1

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
        z = x[:,:64,:]
        x = x[:,64:,:]
        z = token2patch(z)
        x = token2patch(x)

        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) # bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) # bs,c,1,1
        x = x * y.expand_as(x)
        x = patch2token(x)

        y_z = self.gap(z)  # bs,c,1,1
        y_z = y_z.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y_z = self.conv(y_z)  # bs,1,c
        y_z = self.sigmoid(y_z)  # bs,1,c
        y_z = y_z.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        z = z * y_z.expand_as(z)
        z = patch2token((z))
        x = torch.cat((z,x),dim=1)

        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)

        self.adapter_down1 = nn.Linear(768, dim)
        self.adapter_up1 = nn.Linear(dim, 768)
        self.adapter_mid1 = nn.Linear(dim, dim)

        # self.norm = nn.LayerNorm(768)
        self.norm1 = nn.LayerNorm(768)
        self.mix_att1 = Attention_Module(768)
        # self.mix_att2 = Attention_Module(768)
        self.mlp1 = Mlp(768)
        # self.sum = ECAAttention(kernel_size=3)
        # self.sum2 = ECAAttention(kernel_size=3)
        # self.sum = ECAAttention(kernel_size=3)
        self.ro1 = Router(768)
        # self.self_att = Attention_Module2(768)
        # self.ca = ChannelAttentionBlock(768)
        # self.mamba1 = mamba_fusion5(768)
        # self.lpu = LPU(768)
        # self.ldc1 = LDC(768,768)
        # self.ldc2 = LDC(768,768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        nn.init.zeros_(self.adapter_mid1.bias)
        nn.init.zeros_(self.adapter_mid1.weight)
        nn.init.zeros_(self.adapter_down1.weight)
        nn.init.zeros_(self.adapter_down1.bias)
        nn.init.zeros_(self.adapter_up1.weight)
        nn.init.zeros_(self.adapter_up1.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        # self.dim = dim

    def forward(self, x, xi):

        res_x = x
        res_xi = xi
        # ------------ablation 1------------ #

        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        xi_down = self.adapter_down1(xi)
        xi_down = self.adapter_mid1(xi_down)
        xi_down = self.dropout(xi_down)
        xi_up = self.adapter_up1(xi_down)

        mix_x = self.mix_att1(x,xi)
        mix_xi = self.mix_att1(xi,x)

        # ------------------------ #
        # fusion = mix_xi + mix_x
        # fusion = self.self_att(fusion)
        # fusion = self.mlp1(self.norm1(fusion))

        r_x = self.mlp1(self.norm1(mix_x))
        r_xi = self.mlp1(self.norm1(mix_xi))
        # ------------------------ #

        x = r_xi + x_up
        xi = r_x + xi_up
        # ------------ablation 1------------ #


        # ------------ablation 2------------ #
        x = self.ro1(x)
        xi = self.ro1(xi)
        # ------------ablation 2------------ #


        # ------------ablation 3------------ #
        return res_x + xi , res_xi + x

        # ------------ablation 3------------ #

class Self_Attention(nn.Module):
    """
    输入：B N C
    输出：B N C
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


# x = torch.ones(2,320,768)
# m = Bi_direct_adapter(768)
# o = m(x)
# print(o.shape)