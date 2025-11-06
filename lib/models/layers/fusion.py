
from torch import nn
import torch
from torch.nn import init

# from lib.models.layers.CFN import CFN
# from lib.models.layers.adapter_upgraded4 import Attention_Module1
from lib.models.layers.cwt import DecoderCFALayer
from lib.models.layers.fusion_core import fusion_core,GIM
from lib.models.layers.fusion_aff import fusion_aff,GIM
from lib.models.layers.lama_fft1_plus9 import CrossAttention
# from lib.models.layers.z_fusion import ChannelAttentionBlock
from lib.utils.token_utils import patch2token, token2patch
# from lib.models.layers.fusion_mamba import mamba_fusion1, mamba_fusion2, mamba_fusion3, mamba_fusion4,mamba_fusion5
from lib.models.layers.fusion_shaf import DLK,DLKModule,DLKBlock
# from timm.models.vision_transformer import Attention
from lib.models.layers.MA import MutualAttention, MixAttention, LSA, ChannelAttentionBlock, Attention, ChannelAttention, \
    ChannelAttentionBlock, CBAMLayer, ShuffleAttention, AFF, MS_FFN, Router, Router2, MultiScaleDWConv, ECAAttention2, \
    Attention_Module, AFF2, Bi_direct_adapter, ECAAttention, Attention_Module1, CFN, mamba_fusion5, DFF, \
    MutualAttentionNo, SMFFL, Dff_fusion, Bi_direct_adapter1, LSKblock, LDC


# from model.encoders.vmamba import ECAAttention


class Bi_direct_adapter2(nn.Module):
    def __init__(self,in_dim, hidden_dim=384, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(in_dim, hidden_dim)
        self.adapter_up = nn.Linear(hidden_dim, in_dim)
        self.adapter_mid = MS_FFN(hidden_dim)
        # self.adapter_mid = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = hidden_dim

    def forward(self, x):
        # x2 = patch2token(x)
        x = patch2token(x)
        # x = torch.cat((x1, x2), dim=1)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # x_sum = self.linear(x)
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = patch2token(self.adapter_mid(token2patch(x_down)))
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        x_up = token2patch(x_up)
        #print("return adap x", x_up.size())
        return x_up

class SE(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
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

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_sum = self.linear(x)
        x = token2patch(x_sum)
        # B, _, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape) #  ([2, 768, 16, 16])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape) ([2, 768, 1, 1])
        x_out =x * y.expand_as(x)
        return x_out + x1, x_out + x2

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
        # print("q",x.shape)  ([2, 512, 768])
        x_v, x_i = torch.split(x, (N, N,), dim=1)
        # print("x_v, x_i = torch.split(x, (N, N,), dim=1)",x_i.shape) ([2, 256, 768])

        x_v = token2patch(x_v)
        x_i = token2patch(x_i)
        # print("x_i = token2patch(x_i)",x_i.shape) ([2, 768, 16, 16])

        return x_v, x_i

class SE2(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, channel, reduction=16):
        super(SE2, self).__init__()
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

class Mlp(nn.Module):
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
        # x = patch2token(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = token2patch(x)
        return x

class fusion(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.mix_atten = Attention_Module(768)
        self.mlp2 = Mlp(768)
        self.sum = ECAAttention(kernel_size=3)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)





        self.aff = AFF(dim)
        # self.aff2 = AFF(dim)
        self.dff = DFF(dim)
        self.dff2 = DFF(dim)
        # self.mamba1 = mamba_fusion5(dim)
        # self.mamba2 = mamba_fusion5(dim)
        self.attn_sum1 = MutualAttention(dim)
        # self.attn_sum2 = MutualAttention(dim)
        # self.attn_sum3 = MutualAttention(dim)
        # self.attn_sum4 = MutualAttention(dim)
        self.mlp1 = MS_FFN(dim * 2)
        self.mlp2 = MS_FFN(dim * 2)
        # self.norm = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.adapter0 = Bi_direct_adapter(dim)
        self.adapter1 = Bi_direct_adapter(dim)
        self.adapter2 = Bi_direct_adapter(dim)
        # self.adapter3 = MS_FFN(dim)
        # self.adapter4 = MS_FFN(dim)
        # self.adapter3 = Bi_direct_adapter(dim)
        # self.adapter4 = Bi_direct_adapter(dim)
        # self.act = nn.Sigmoid()
        # self.act1 = nn.Sigmoid()

        # self.df1 = Dff_fusion(dim)
        # self.df2 = Dff_fusion(dim)

        # self.se1 = SE2(dim)
        # self.se2 = SE2(dim)
        # self.smffl1 = LDC(dim,dim)
        # self.smffl2 = SMFFL(dim,dim)
        # self.ldc = LDC(dim,dim)
        # self.glo = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim),
        # )

        self.lpu = LPU(dim)

        self.ca1 = ChannelAttentionBlock(dim)
        # self.router2 = ChannelAttentionBlock(dim)


        # self.local_att1 = nn.Sequential(
        #     nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim)
        # )
        # self.local_att2 = nn.Sequential(
        #     nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim)
        # )

        # self.ms1 = MultiScaleDWConv(dim)
        # self.ms2 = MultiScaleDWConv(dim)


        # self.stb1 = SGBlock()
        # self.stb2 = SGBlock()
        # self.fc1 = DecoderCFALayer(dim)
        # self.fc2 = DecoderCFALayer(dim)
        # self.fc2 = nn.Sequential(
        #     nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim),
        # )
        # self.CA1= ChannelAttention(dim)
        # self.CA2 = ChannelAttention(dim)
        # self.mamba5 = mamba_fusion5(dim)
        # self.ro2 = Router2(dim)
        # self.ro2_1 = Router2(dim)
        # self.cfn = CFN(dim, dim)
        # self.fuse = ECAAttention(dim)
        # self.fuse2 = ECAAttention2(dim)
        #
        # self.t_fusion = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.linear1 = nn.Linear(dim * 2,dim)
        # self.linear2 = nn.Linear(dim * 2,dim)
        # # self.router1 = Router(dim)
        # self.router2 = Router(dim)
        # self.router3 = Router2(dim)
        # self.router4 = Router2(dim)
        # self.aff = AFF(dim)
        # self.aff2 = AFF(dim)
        # self.aff_1 = AFF2(dim,16,16)
        # self.aff_2 = AFF2(dim,16,16)
        # self.gim1 = GIM(dim)
        # self.s2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        #     # nn.BatchNorm2d(dim),
        # )
        # self.se3 = SE2(dim)
        # self.ms1 = MultiScaleDWConv(dim)
        # self.ms2 = MultiScaleDWConv(dim)
        # self.mamba = mamba_fusion1(dim)
        # self.dlk = DLKBlock(dim)
        # self.mamba2 = mamba_fusion2(dim)
        # self.mamba3 = mamba_fusion2(dim)
        # self.mamba2_x = mamba_fusion2(dim)
        # self.mamba3 = mamba_fusion3(dim)
        # self.mamba4 = mamba_fusion4(dim)
        # self.norm= nn.LayerNorm(dim // 2)
        # self.attn_1 = ChannelAttention(dim)
        # self.attn_2 = ChannelAttention(dim)
        # self.x_fusion = ChannelAttention(dim)
        # self.attn1 = CrossAttention(dim // 2,dim // 2,dim // 2,dim // 2)
        # self.mix = MixAttention(dim)
        # self.mix_attn2 = MixAttention(dim // 2)
        # self.attn1 = CrossAttention(dim,dim,dim,dim)
        # self.attn2 = CrossAttention(dim,dim,dim,dim)
        # self.down = nn.Linear(dim,dim // 2)
        # self.down2 = nn.Linear(dim,dim // 2)
        # self.up = nn.Linear(dim // 2,dim)
        # self.linear = nn.Linear(dim * 2, dim)
        # self.op00 = Attention(dim)
        # self.op01 = Attention(dim)
        # self.op1 = Attention(dim)
        # self.op2 = Attention(dim)
        # self.op3 = Attention(dim)


        # self.oplus1 = Attention_Module1(dim)
        # self.oplus2 = Attention_Module1(dim)
        # self.fusion1 = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.fusion2 = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.op1_i = Attention(dim)
        # self.op2_i = Attention(dim)
        # self.op2 = ChannelAttention(dim // 2)
        # self.eca = ECAAttention2(dim)
        # self.eca2 = ECAAttention2(dim)
        # self.op3 = ShuffleAttention(dim)
        # self.op3 = ShuffleAttention(dim)
        # self.op4 = LSA(dim,dim)
        # self.op5 = LSA(dim,dim)
        # self.adapter0 = Bi_direct_adapter(dim)
        # self.adapter02 = Bi_direct_adapter(dim)
        # self.adapter01 = Bi_direct_adapter(dim)
        # self.adapter1 = Bi_direct_adapter2(dim)
        # self.adapter02 = Bi_direct_adapter(dim)
        # self.adapter2 = Bi_direct_adapter2(dim)
        # self.adapter5 = Bi_direct_adapter(dim)
        # self.adapter6 = Bi_direct_adapter(dim)
        # self.linear = nn.Linear(dim * 2,dim)
        # self.linear2 = nn.Linear(dim * 2,dim)
        # self.adapter_down = nn.Linear(768, 8)
        # self.adapter_up = nn.Linear(8, 768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.weight)
        # nn.init.zeros_(self.adapter_up.bias)
        # self.down = nn.Linear(dim,dim // 2)
        # self.up = nn.Linear(dim // 2,dim)
        # self.up2 = nn.Linear(dim // 2,dim)
        # self.silu = nn.SiLU(inplace=True)
        # self.act = nn.ReLU(inplace=True)
        # self.lsa_i = MS_FFN(dim,16,166)
        # self.lsa = MS_FFN(dim,8,8)
        # self.mlp1 = Mlp(dim)
        # self.mlp2 = Mlp(dim // 2)
        # self.norm =  nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)

        # self.mamba = mamba_fusion5(self.channels)

        # self.gim1 = GIM(dim)
        # self.gim2 = GIM(dim)

    def forward(self, x, xi):

        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        xi_down = self.adapter_down(xi)
        xi_down = self.adapter_mid(xi_down)
        xi_down = self.dropout(xi_down)
        xi_up = self.adapter_up(xi_down)

        # x = x + self.adapter3(x1) + self.adapter3(x2)
        # xi = xi + self.adapter4(xi1) + self.adapter4(xi2)
        # x = torch.cat((x, x1, x2),dim=2)
        # xi = torch.cat((xi, xi1, xi2),dim=2)

        # x = self.down1(x)
        # xi = self.down2(xi)

        # x_down = self.adapter_down(x)
        # raw = x_down
        # # ---------------ablation ------------ #
        # x_down = self.stb1(x_down)
        # x_down = self.stb2(x_down)
        # # ---------------ablation ------------ #
        # x = raw + x_down
        # x = self.adapter_up(x)
        #
        # xi_down = self.adapter_down(xi)
        # raw = xi_down
        # # ---------------ablation ------------ #
        # xi_down = self.stb1(xi_down)
        # xi = self.stb2(xi_down)
        # # ---------------ablation ------------ #
        # xi = raw + xi      # x = self.norm(x)
        # xi = self.adapter_up(xi)
        # xi = self.norm(xi)

        z_v = x[:,:64,:]
        x_v = x[:,64:,:]
        z_i = xi[:,:64,:]
        x_i = xi[:,64:,:]


        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)


        z_v_res, z_i_res = z_v, z_i
        x_v_res, x_i_res = x_v, x_i

        res_z_fusion = self.dff(z_v, z_i)
        z_v = z_v + res_z_fusion
        z_i = z_i + res_z_fusion

        res_x_fusion = self.dff(x_v, x_i)
        x_v = x_v + res_x_fusion
        x_i = x_i + res_x_fusion

        # z_v_fusion_tmp = self.router1(z_v) + self.local_att1(z_v)
        # z_i_fusion_tmp = self.router1(z_i) + self.local_att1(z_i)
        # res_z_fusion = z_v_fusion_tmp + z_i_fusion_tmp

        # x_v_fusion_tmp = self.router2(x_v) + self.local_att2(x_v)
        # x_i_fusion_tmp = self.router2(x_i) + self.local_att2(x_i)
        # res_x_fusion = x_i_fusion_tmp + x_v_fusion_tmp


        z_v, z_i = self.mlp1(z_v, z_i)
        x_v, x_i = self.mlp2(x_v, x_i)

        z_v = self.lpu(z_v)
        z_i = self.lpu(z_i)
        x_v = self.lpu(x_v)
        x_i = self.lpu(x_i)

        zi_fusion = self.attn_sum1(z_v, z_i)
        zv_fusion = self.attn_sum1(z_i, z_v)
        # res_z_v_fusion = self.attn_sum1(z_v, z_i)
        # res_z_i_fusion = self.attn_sum1(x_v, x_i)
        zi_fusion = self.adapter1(zi_fusion)
        zv_fusion = self.adapter1(zv_fusion)


        z_fusion = self.ca1(zv_fusion + zi_fusion)
        # z_v_fusion = self.adapter1(res_z_v_fusion)
        # z_i_fusion = self.adapter1(res_z_i_fusion)
        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # res1 = z_v_fusion
        # z_v_fusion = self.mamba1(z_v_fusion)
        # z_v_fusion = self.router1(z_v_fusion) + res1
        #
        # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # res2 = z_i_fusion
        # z_i_fusion = self.mamba1(z_i_fusion)
        # z_i_fusion = self.router1(z_i_fusion) + res2

        # z_fusion = self.mamba1(z_v_fusion + z_i_fusion)
        # z_fusion = self.router1(z_fusion)

        # z_v_fusion = z_fusion + z_v_fusion
        # z_v_fusion = self.router1(z_v_fusion)
        # z_i_fusion = z_fusion + z_i_fusion
        # z_i_fusion = self.router1(z_i_fusion)



        # z_v_fusion = z_v_fusion * self.df1(z_i)
        # z_v_fusion = self.mamba1(z_v_fusion)
        # z_v_fusion = self.act(self.adapter1(z_v_fusion))
        # z_v_fusion = self.smffl1(z_v_fusion)

        # z_v_fusion = self.mlp1(z_v_fusion) + self.act(self.adapter3(z_v_fusion)) + self.mamba1(z_v_fusion)
        # z_v_fusion = self.mlp1(z_v_fusion) + self.mamba1(z_v_fusion)
        # z_v_fusion = self.mamba1(z_v_fusion)

        # z_i_fusion = self.attn_sum2(res_z_fusion, res_z_i_fusion)
        # z_i_fusion = self.mamba1(z_i_fusion)
        # z_v_fusion = self.mlp1(z_i_fusion) + self.act(self.adapter3(z_i_fusion)) + self.mamba1(z_i_fusion)
        # z_i_fusion = self.mlp1(z_i_fusion) + self.mamba1(z_v_fusion)

        # z_i_fusion = z_i_fusion * self.df1(z_v)
        # z_i_fusion = self.mamba1(z_i_fusion)
        # z_i_fusion = self.act(self.adapter1(z_i_fusion))
        # z_i_fusion = self.smffl1(z_i_fusion)

        xi_fusion = self.attn_sum1(x_v, x_i)
        xv_fusion = self.attn_sum1(x_i, x_v)
        # res_x_v_fusion = self.attn_sum1(res_x_fusion, x_v)
        # res_x_i_fusion = self.attn_sum1(res_x_fusion, x_i)

        xi_fusion = self.adapter1(xi_fusion)
        xv_fusion = self.adapter1(xv_fusion)


        x_fusion = self.ca1(xv_fusion + xi_fusion)

        # x_v_fusion = self.adapter2(res_x_v_fusion)
        # x_i_fusion = self.adapter2(res_x_i_fusion)
        # x0_v_fusion = self.attn_sum1(x_v, x_i)
        # x1_v_fusion = self.attn_sum1(x_v, x1_i)
        # x2_v_fusion = self.attn_sum1(x_v, x2_i)
        #
        # x_i_fusion = self.attn_sum1(z_i, z_v)
        # x_i_fusion = self.attn_sum1(z_i, z1_v)
        # x_i_fusion = self.attn_sum1(z_i, z2_v)
        #
        # x_v_fusion = x_v_fusion + x_v_fusion + x_v_fusion
        # x_i_fusion = x_i_fusion + x_i_fusion + x_i_fusion

        # x_v_fusion = token2patch(self.norm2(patch2token(x_v_fusion)))
        # res3 = x_v_fusion
        # x_v_fusion = self.mamba2(x_v_fusion)
        # x_v_fusion = self.router2(x_v_fusion) + res3
        #


        # x_i_fusion = token2patch(self.norm2(patch2token(x_i_fusion)))
        # res4 = x_i_fusion
        # x_i_fusion = self.mamba2(x_i_fusion)
        # x_i_fusion = self.router2(x_i_fusion) + res4

        # x_fusion = self.mamba2(x_v_fusion + x_i_fusion)
        # x_fusion = self.router2(x_fusion)

        # x_v_fusion = x_fusion + x_v_fusion
        # x_v_fusion = self.router2(x_v_fusion)
        # x_i_fusion = x_fusion + x_i_fusion
        # x_i_fusion = self.router2(x_i_fusion)
        # x_v_fusion = self.mlp2(x_v_fusion) + self.mamba2(x_v_fusion)
        # x_i_fusion = x_i_fusion * self.df2(x_v)
        # x_i_fusion = self.mamba2(x_i_fusion)
        # x_i_fusion = self.act1(self.adapter2(x_i_fusion))
        # x_i_fusion = self.act1(self.adapter2(x_i_fusion))
        # x_i_fusion = self.smffl2(x_i_fusion)

        # x_i_fusion = self.attn_sum4(res_x_fusion, x_i_fusion)
        # x_i_fusion = self.mamba2(x_i_fusion)
        # x_i_fusion = self.mlp2(x_i_fusion) + self.act1(self.adapter4(x_i_fusion)) + self.mamba2(x_i_fusion)
        # x_i_fusion = self.mlp2(x_i_fusion) + self.mamba2(x_i_fusion)
        # x_i_fusion = self.mamba2(x_i_fusion)
        # x_v_fusion = self.attn_sum4(res_x_fusion, x_v_fusion)
        # x_v_fusion = x_v_fusion * self.df2(x_i)
        # x_v_fusion = self.mamba2(x_v_fusion)
        # x_v_fusion = self.act1(self.adapter2(x_v_fusion))
        # x_v_fusion = self.smffl2(x_v_fusion)
        # x_v_fusion = self.mlp2(x_v_fusion) + self.act1(self.adapter4(x_v_fusion)) + self.mamba2(x_v_fusion)

        z_v, z_i = z_fusion + z_v_res, z_fusion + z_i_res
        x_v, x_i = x_fusion + x_v_res, x_fusion + x_i_res

        # z_v_fusion = self.mlp1(z_v_fusion) + self.mamba1(z_v_fusion)

        # z_v_fusion = self.mlp1(z_v_fusion) + self.act(self.adapter3(res_z_v_fusion))
        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # res_z_v_fusion = z_v_fusion
        # z_v_fusion = self.mlp1(z_v_fusion)
        # z_v_fusion = z_v_fusion + self.mlp1(z_v_fusion)
        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        #


        # z_v_fusion = self.oplus1(z_i_fusion)
        # z_i_fusion = self.oplus1(z_i_fusion)
        # z_i_fusion = self.adapter3(z_i_fusion)

        # z_i_fusion = self.mlp1(z_i_fusion) + self.act(self.adapter3(res_z_i_fusion))

        # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # z_v_fusion = self.oplus1(z_v_fusion)
        # z_i_fusion = self.oplus1(z_i_fusion)
        # z_fusion = torch.cat((z_v_fusion, z_i_fusion), dim=1)
        # z_fusion = self.fusion1(patch2token(z_fusion))
        # z_fusion = z_v_fusion + z_i_fusion

        # z = z.flatten(2).transpose(1, 2).contiguous()
        # z_sum = self.linear(z)
        # z = token2patch(z_sum)

        # x_v = self.op2(x_v)
        # x_i = self.op2(x_i)
        # x_fusion = self.op1(x_fusion)

        # x_v_fusion = self.mlp2(x_v_fusion) + self.mamba2(x_v_fusion)

        # x_v_fusion = self.mlp2(x_v_fusion) + self.act(self.adapter4(res_x_v_fusion))
        # x_v_fusion = token2patch(self.norm(patch2token(x_v_fusion)))
        # res_x_v_fusion = x_v_fusion
        # x_v_fusion = self.mlp1(x_v_fusion)
        # x_v_fusion = x_v_fusion + self.mlp1(x_v_fusion)
        # x_v_fusion = token2patch(self.norm(patch2token(x_v_fusion)))
        #


        # z_v = patch2token(z_v)
        # x_v = patch2token(x_v)
        # z_i = patch2token(z_i)
        # x_i = patch2token(x_i)
        #
        # xv = torch.cat((z_v,x_v),dim=1)
        # xv = self.act(self.down(xv))
        # xv = xv + self.mix(xv)
        # xv = self.up(xv)
        # z_v = xv[:,:64,:]
        # x_v = xv[:,64:,:]
        #
        # xi = torch.cat((z_i, x_i), dim=1)
        # xi = self.act(self.down(xi))
        # xi = xi + self.mix(xi)
        # xi = self.up(xi)
        # z_i = xi[:, :64, :]
        # x_i = xi[:, 64:, :]
        #
        # z_v = token2patch(z_v)
        # x_v = token2patch(x_v)
        # z_i = token2patch(z_i)
        # x_i = token2patch(x_i)
        # z_v, z_i = self.mamba5(z_v, z_i)
        # x_v, x_i = self.mamba5(x_v, x_i)
        # z_v, z_i = self.gim1(z_v, z_i)
        # x_v, x_i = self.gim2(x_v, x_i)

        # z_fusion = self.attn_1(z_v, z_i)
        # x_fusion = self.attn_1(x_v, x_i)
        # z_v, z_i = self.aff(z_v, z_i)
        # x_v, x_i = self.aff2(x_v, x_i)
        # z_v, z_i = self.router3(z_v, z_i)
        # x_v, x_i = self.router3(x_v, x_i)

        # z_v = self.router(z_v)
        # x_v = self.router(x_v)
        # z_i = self.router1(z_i)
        # x_i = self.router1(x_i)
        # z_v = self.attn1(z_v, z_i)
        # z_i = self.attn1(z_i, z_v)
        # x_v = self.attn2(x_v, x_i)
        # x_i = self.attn2(x_i, x_v)
        # z_v, z_i = self.gim1(z_v, z_i)
        # x_v, x_i = self.gim2(x_v, x_i)

        # z_v, z_i = self.attn_1(z_v, z_i)
        # x_v, x_i = self.attn_1(x_v, x_i)
        # z_v, z_i = self.gim1(z_v, z_i)
        # x_v, x_i = self.gim2(x_v, x_i)
        # z_v, z_i = self.lsa(z_v, z_i)
        # x_v, x_i = self.lsa_i(x_v, x_i)

        # z_v, z_i = self.ro2(z_v, z_i)
        # x_v, x_i = self.ro2_1(x_v, x_i)

        # zv_fusion = self.attn_1(z_v, z_i)
        # zi_fusion = self.attn_1(z_v, z_i) + self.se2(z_i_res)
        # zi_fusion = self.op00(zi_fusion)

        # xv_fusion = self.attn_1(x_v, x_i) + self.se2(x_v_res)
        # xv_fusion = self.op01(xv_fusion)
        # xi_fusion = self.attn_1(x_v, x_i) + self.se2(x_i_res)
        # xi_fusion = self.op01(xi_fusion)
        # x_v, x_i = self.attn_1(x_v, x_i)
        # z_v, z_i = self.se(z_v, z_i)
        # x_v, x_i = self.se(x_v, x_i)

        # z_v, z_i = self.fuse(z_v, z_i)
        # x_v, x_i = self.fuse2(x_v, x_i)
        # z_v, z_i = self.gim1(z_v, z_i)
        # x_v, x_i = self.gim2(x_v, x_i)


        # z_v, z_i = self.se(z_v, z_i)
        # x_v, x_i = self.se(x_v, x_i)
        # z_fusion = self.mamba2(z_v, z_i)
        # x_fusion = self.mamba2(x_v, x_i)
        # z_fusion = self.mamba2(z_v, z_i)
        # x_fusion = self.mamba2(x_v, x_i)
        # print(x_fusion.shape)
        # z_fusion = self.fuse(z_fusion)
        # x_fusion = self.fuse(x_fusion)


        # z_fusion = self.adapter1(z_fusion) + self.mlp1(self.norm(patch2token(z_fusion)))
        # x_fusion = self.adapter2(x_fusion) + self.mlp1(self.norm(patch2token(z_fusion)))
        # z_fusion = self.mlp1(patch2token(z_fusion)) + z_fusion
        # x_fusion = self.mlp2(patch2token(x_fusion)) + x_fusion
        # z_fusion = self.op3(z_fusion)
        # x_fusion = self.op3(x_fusion)
        # z_fusion = self.adapter1(z_fusion) + z_fusion
        # x_fusion = self.adapter2(x_fusion) + x_fusion


        # x2_fusion = self.mamba2(x_i, x_v)



        # z_fusion = self.adapter1(z_fusion)
        # x_fusion = self.adapter1(x_fusion)

        # z_fusion = self.lsa(z_fusion)
        # x_fusion = self.lsa_i(x_fusion)

        # z_v, z_i = self.eca(z_v, z_i)
        # x_v, x_i = self.eca2(x_v, x_i)

        # z1_fusion = self.router3(z1_fusion)
        # z_v = self.router1(z_v)
        # z_i = self.router1(z_i)
        # x_v = self.router2(x_v)
        # x_i = self.router2(x_i)
        # z_fusion = self.aff_2(z_v, z_i)
        # x_fusion = self.aff_2(x_v, x_i)
        # z_v = self.adapter3(z_v)
        # z_i = self.adapter3(z_i)

        # z_fusion = self.attn_sum1(z_v, z_fusion) + self.attn_sum1(z_i, z_fusion)

        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # z_v_fusion = z_v_fusion + self.mlp1(z_v_fusion)
        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))

        # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # z_i_fusion = z_i_fusion + self.mlp1(z_i_fusion)
        # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # z1_fusion = torch.cat((z_v_fusion,z_i_fusion),dim=1)
        # z1_fusion = z1_fusion.flatten(2).transpose(1, 2).contiguous()
        # z1_fusion = self.linear1(z1_fusion)
        # z1_fusion = token2patch(z1_fusion)

        # z1_fusion = self.op2(z1_fusion)
        # z1_fusion = token2patch(self.norm(patch2token(z1_fusion)))
        # z1_fusion = z1_fusion + self.mlp1(z1_fusion)
        # z1_fusion = token2patch(self.norm(patch2token(z1_fusion)))

        # x1_fusion = self.router3(x1_fusion)
        # x_v = self.router2(x_v)
        # x_i = self.router2(x_i)
        # x_v = self.op1_i(x_v)
        # x_i = self.op1_i(x_i)
        # x_v = self.adapter4(x_v)
        # x_i = self.adapter4(x_i)
        # z_v = self.op1(z_v)
        # z_i = self.op1(z_i)
        # x_v = self.op2(x_v)
        # x_i = self.op2(x_i)

        # res_z_fusion = self.adapter3(res_z_fusion)
        # res_x_fusion = self.adapter4(res_x_fusion)

        # z_v = self.op1(z_v)
        # z_i = self.op1(z_i)


        # res_z_v_fusion = self.attn_sum1(z_v, res_z_fusion)
        # z_v_fusion = self.attn_sum2(res_z_fusion, res_z_v_fusion)
        # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # z_v_fusion = self.mlp1(z_v_fusion) + self.act(self.adapter3(z_v_fusion))
        # # z_v_fusion = self.mlp1(z_v_fusion) + self.mamba1(z_v_fusion)
        #
        # # z_v_fusion = self.mlp1(z_v_fusion) + self.act(self.adapter3(res_z_v_fusion))
        # # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # # res_z_v_fusion = z_v_fusion
        # # z_v_fusion = self.mlp1(z_v_fusion)
        # # z_v_fusion = z_v_fusion + self.mlp1(z_v_fusion)
        # # z_v_fusion = token2patch(self.norm(patch2token(z_v_fusion)))
        # #
        #
        # res_z_i_fusion = self.attn_sum1(z_i, res_z_fusion)
        # z_i_fusion = self.attn_sum2(res_z_fusion, res_z_i_fusion)
        # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # z_i_fusion = self.mlp1(z_i_fusion) + self.act(self.adapter3(z_v_fusion))
        #
        # # z_v_fusion = self.oplus1(z_i_fusion)
        # # z_i_fusion = self.oplus1(z_i_fusion)
        # # z_i_fusion = self.adapter3(z_i_fusion)
        #
        # # z_i_fusion = self.mlp1(z_i_fusion) + self.act(self.adapter3(res_z_i_fusion))
        #
        # # z_i_fusion = token2patch(self.norm(patch2token(z_i_fusion)))
        # # z_v_fusion = self.oplus1(z_v_fusion)
        # # z_i_fusion = self.oplus1(z_i_fusion)
        # # z_fusion = torch.cat((z_v_fusion, z_i_fusion), dim=1)
        # # z_fusion = self.fusion1(patch2token(z_fusion))
        # # z_fusion = z_v_fusion + z_i_fusion
        #
        # # z = z.flatten(2).transpose(1, 2).contiguous()
        # # z_sum = self.linear(z)
        # # z = token2patch(z_sum)
        #
        # # x_v = self.op2(x_v)
        # # x_i = self.op2(x_i)
        # # x_fusion = self.op1(x_fusion)
        #
        # res_x_v_fusion = self.attn_sum3(x_v, res_x_fusion)
        # x_v_fusion = self.attn_sum4(res_x_fusion, res_x_v_fusion)
        # x_v_fusion = token2patch(self.norm2(patch2token(x_v_fusion)))
        # x_v_fusion = self.mlp2(x_v_fusion) + self.act1(self.adapter4(x_v_fusion))
        # # x_v_fusion = self.mlp2(x_v_fusion) + self.mamba2(x_v_fusion)
        #
        # # x_v_fusion = self.mlp2(x_v_fusion) + self.act(self.adapter4(res_x_v_fusion))
        # # x_v_fusion = token2patch(self.norm(patch2token(x_v_fusion)))
        # # res_x_v_fusion = x_v_fusion
        # # x_v_fusion = self.mlp1(x_v_fusion)
        # # x_v_fusion = x_v_fusion + self.mlp1(x_v_fusion)
        # # x_v_fusion = token2patch(self.norm(patch2token(x_v_fusion)))
        # #
        #
        # res_x_i_fusion = self.attn_sum3(x_i, res_x_fusion)
        # x_i_fusion = self.attn_sum4(res_x_fusion, res_x_i_fusion)
        # x_i_fusion = token2patch(self.norm2(patch2token(x_i_fusion)))
        # x_i_fusion = self.mlp2(x_i_fusion) + self.act1(self.adapter4(x_i_fusion))
        # x_i_fusion = self.mlp2(x_i_fusion) + self.mamba2(x_i_fusion)

        # x_v_fusion = self.oplus2(x_v_fusion)
        # x_i_fusion = self.oplus2(x_i_fusion)



        # x_fusion = torch.cat((x_v_fusion, x_i_fusion), dim=1)
        # x_fusion = self.fusion1(patch2token(x_fusion))
        # x_fusion = x_v_fusion + x_i_fusion

        # x_i_fusion = self.mlp2(x_i_fusion) + self.act(self.adapter4(res_x_i_fusion))
        # x_v_fusion = self.oplus2(x_v_fusion)
        # x_i_fusion = self.oplus2(x_i_fusion)



        # x_v_fusion = self.op2(x_v_fusion)
        # x_i_fusion = self.op2(x_i_fusion)

        # x = torch.cat((x_v_fusion, x_i_fusion), dim=1)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # x_sum = self.linear2(x)
        # x = token2patch(x_sum)
        # x_i_fusion = token2patch(self.norm(patch2token(x_i_fusion)))
        # x_i_fusion = self.mlp1(x_i_fusion)
        # x_i_fusion = x_i_fusion + self.mlp1(x_i_fusion)
        # x_i_fusion = token2patch(self.norm(patch2token(x_i_fusion)))

        # x1_fusion = torch.cat((x_v_fusion,x_i_fusion),dim=1)
        # x1_fusion = x1_fusion.flatten(2).transpose(1, 2).contiguous()
        # x1_fusion = self.linear2(x1_fusion)
        # x1_fusion = token2patch(x1_fusion)
        # x1_fusion = res_x_fusion +  token2patch(x1_fusion)
        # x1_fusion = self.fren2(x_v_fusion, x_i_fusion)
        # x1_fusion = x_v_fusion + x_i_fusion

        # x1_fusion = x1_fusion + self.op2(x1_fusion)
        # x1_fusion = self.router2(x1_fusion)
        # x1_fusion = token2patch(self.norm(patch2token(x1_fusion)))
        # x1_fusion = x1_fusion + self.mlp2(x1_fusion)
        # x1_fusion = token2patch(self.norm(patch2token(x1_fusion)))


        # z1_fusion = self.adapter1(z1_fusion)
        # x1_fusion = self.adapter2(x1_fusion)
        # z_fusion = z_fusion + self.lsa(z_fusion)
        # x_fusion = x_fusion + self.lsa_i(x_fusion)


        # z_fusion = z_fusion + self.mlp1(self.norm(patch2token(z_fusion)))
        # x_fusion = x_fusion + self.mlp2(self.norm(patch2token(x_fusion)))
        # z1_fusion = z0_fusion + z1_fusion
        # x1_fusion = x0_fusion + x1_fusion
        # B,C,H1,W1 = z1_fusion.shape
        # z1_fusion = self.cfn(z1_fusion, H1, W1)
        # B,C,H2,W2 = x1_fusion.shape
        # x1_fusion = self.cfn(x1_fusion, H2, W2)

        # z1_fusion = self.se2(z1_fusion)
        # x1_fusion = self.se2(x1_fusion)


        z_v = patch2token(z_v)
        x_v = patch2token(x_v)
        z_i = patch2token(z_i)
        x_i = patch2token(x_i)


        x_v = torch.cat((z_v, x_v), dim=1)
        x_i = torch.cat((z_i, x_i), dim=1)

        return x_v, x_i


class LPU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.conv(x) + x


# x = torch.ones(2,320,768)
# m = fusion()
# o_v,o_i = m(x,x)
# print(o_v.shape)
from functools import partial
from turtle import forward

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from lib.models.layers.CASTBlock import CASTBlock
# from lib.models.layers.fusion_mamba import mamba_fusion1
# from lib.utils.token_utils import token2patch, patch2token
#
#
# class fusion(nn.Module):
#     def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#
#         self.t_fusion = mamba_fusion1(dim)
#
#         self.ca_s2t_v2f = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#         self.ca_t2s_f2i = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#         self.ca_s2t_i2f = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#         self.ca_t2s_f2v = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#         self.ca_t2t_f2v = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#         self.ca_t2t_f2i = CASTBlock(
#             dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
#             attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
#         )
#
#     def forward(self, x_v, x_i):
#         lens_z = 64
#         # x_v: [B, N, C], N = 320
#         # x_i: [B, N, C], N = 320
#         # fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
#         z_v = x_v[:,:64,:]
#         z_i = x_i[:, :64, :]
#         z_v = token2patch(z_v)
#         z_i = token2patch(z_i)
#
#         fused_t = self.t_fusion(z_v,z_i)  # [B, 64, C]
#
#         fused_t = patch2token(fused_t)
#
#         fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]
#         temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, lens_z:, :]
#         fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
#         temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]
#         x_v[:, lens_z:, :] = temp_x_v
#         x_i[:, lens_z:, :] = temp_x_i
#         x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
#         x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
#
#         return x_v, x_i

# def get_z(x):
#     return x[:,64:,:]
#
# def get_x(x):
#     return x[:,:64,:]