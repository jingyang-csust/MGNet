import torch
import torch.nn as nn
import torch.nn.init as init


class SE(nn.Module):
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

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape) #  ([2, 768, 16, 16])
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        # print(y.shape)  # ([2, 768, 1, 1])
        y = y.view(b, c)
        # print(y.shape)  #  ([2, 768])
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape) ([2, 768, 1, 1])
        # print(y.expand_as(x).shape)  # ([2, 768, 16, 16])
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2).contiguous()

x = torch.ones([2,256,768])
model = SE(768)
xo = model(x,16,16)
print(xo.shape) #  ([2, 256, 768])
