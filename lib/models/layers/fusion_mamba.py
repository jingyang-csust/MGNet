import torch
from torch import nn

from lib.models.layers.utils_mamba import CrossMambaFusionBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.utils.token_utils import patch2token, token2patch
from model.encoders.vmamba import ConcatMambaFusionBlock, CVSSDecoderBlock, VSSBlock


class mamba_fusion1(nn.Module):
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 dims=768,
                 ape=False,
                 drop_path_rate=0.2,):
        super().__init__()

        # self.down = nn.Linear(dims,dims // 2)
        # self.up = nn.Linear(dims // 2,dims)

        self.cross_mamba = CrossMambaFusionBlock(
                hidden_dim=dims,
                mlp_ratio=0.0,
                d_state=4,
            )

        self.channel_attn_mamba = ConcatMambaFusionBlock(
                hidden_dim=dims,
                mlp_ratio=0.0,
                d_state=4,
            )

    def forward_features(self, x1, x2):
        """
        input: B x C x H x W
        output: B x C x H x W
        """
        # x1 = patch2token(x1)
        # x2 = patch2token(x2)
        # x1 = self.down(x1)
        # x2 = self.down(x2)
        # x1 = token2patch(x1)
        # x2 = token2patch(x2)
        cross_x1, cross_x2 = self.cross_mamba(x1.permute(0, 2, 3, 1).contiguous(),
                                                 x2.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
        # cross_x1 = patch2token(cross_x1.permute(0, 3, 1, 2).contiguous())
        # cross_x2 = patch2token(cross_x2.permute(0, 3, 1, 2).contiguous())
        # cross_x1 = self.up(cross_x1)
        # cross_x2 = self.up(cross_x2)
        # cross_x1 = token2patch(cross_x1).permute(0, 2, 3, 1).contiguous()
        # cross_x2 = token2patch(cross_x2).permute(0, 2, 3, 1).contiguous()
        x_fuse = self.channel_attn_mamba(cross_x1, cross_x2).permute(0, 3, 1, 2).contiguous()

        return x_fuse

    def forward(self, x1, x2):
        x_fuse = self.forward_features(x1, x2)
        return x_fuse

class mamba_fusion2(nn.Module):
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 dims=768,
                 ape=False,
                 drop_path_rate=0.2,):
        super().__init__()

        self.ape = ape

        self.channel_attn_mamba = ConcatMambaFusionBlock(
            hidden_dim=dims,
            mlp_ratio=0.0,
            d_state=4,
            state = 1
        )

    def forward_features(self, x1, x2):
        """
        input: B x C x H x W
        output: B x C x H x W
        """
        # x1 = token2patch(x1)
        # x2 = token2patch(x2)
        x_fuse = self.channel_attn_mamba(x1.permute(0, 2, 3, 1).contiguous(), x2.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        return x_fuse

    def forward(self, x1, x2):
        x_fuse = self.forward_features(x1, x2)

        return x_fuse

class mamba_fusion3(nn.Module):
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 dims=768,
                 ape=False,
                 drop_path_rate=0.2,):
        super().__init__()

        self.ape = ape

        self.cross_mamba = CrossMambaFusionBlock(
            hidden_dim=dims,
            mlp_ratio=0.0,
            d_state=4,
            state=1
        )

    def forward_features(self, x1, x2):
        """
        input: B x C x H x W
        output: B x C x H x W
        """
        out = self.cross_mamba(x1.permute(0, 2, 3, 1).contiguous(),
                                              x2.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
        # print(out.shape)
        return out

    def forward(self, x1, x2):
        out = self.forward_features(x1, x2)
        # print("hello")
        return out

class mamba_fusion4(nn.Module):
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

        self.channel_attn_mamba = CVSSDecoderBlock(
                hidden_dim=dim,
                drop_path=0.,
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

    def forward_features(self, x):
        """
        input: B x C x H x W
        output: B x C x H x W
        """
        x_fuse = self.channel_attn_mamba(x.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
        return x_fuse

    def forward(self, x):
        x_fuse = self.forward_features(x)
        return x_fuse

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
    def forward_features(self, x1, x2, x3):
        """
        input:  B x N x C
        output: B x N x C
        """
        # x1 = patch2token(x1)
        # x2 = patch2token(x2)
        # x = torch.cat((x1,x2),dim=2)
        # x_fuse = self.fusion(x)
        # x_fuse = token2patch(x_fuse)
        x_fuse = self.vss(x1, x2, x3)
        return x_fuse

    def forward(self, x1, x2, x3):
        x1 = patch2token(x1)
        x2 = patch2token(x2)
        x3 = patch2token(x3)
        # x2 = patch2token(x2)
        # x = torch.cat((x1,x2),dim=2)
        # x_fuse = self.fusion(x)
        out = self.forward_features(x1, x2, x3)
        out = token2patch(out)
        # x1_out = token2patch(x1 + x_fuse)
        # x2_out = token2patch(x2 + x_fuse)
        return out

# x = torch.ones(2,768,16,16)
# m = mamba_fusion5(768)
# o_v = m(x,x)
# print(o_v.shape)