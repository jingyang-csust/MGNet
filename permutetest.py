import torch
from torch import nn


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=20, w=20, SC=768, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 1x1 convolution and GELU activation
        self.conv1x1_real = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1_imag = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()


    def forward(self, x):
        x = x.permute(0,3,2,1)
        B, a, b, C = x.shape  # ([2, 16, 16, 8])
        # if spatial_size is None:
        #     a = b = int(math.sqrt(n))
        # else:
        #     a, b = spatial_size
        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        # print("ggg:",x.shape)
        # Apply FFT
        print(x.shape)
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        print(x_fft.shape)

        # Apply 1x1 convolution and GELU activation
        x_fft_real = x_fft.real
        print("------++++++",x_fft_real.shape)
        x_fft_real = x_fft_real.permute(0, 3, 1, 2)
        x_fft_imag = x_fft.imag.permute(0, 3, 1, 2)
        print(x_fft_imag.shape)

        # Apply 1x1 convolution to real and imaginary parts separately
        x_fft_real = self.conv1x1_real(x_fft_real)
        x_fft_imag = self.conv1x1_imag(x_fft_imag)
        x_fft_real = self.act(self.norm1(x_fft_real))
        x_fft_imag = self.act(self.norm2(x_fft_imag))
        x_fft = torch.complex(x_fft_real, x_fft_imag).permute(0, 2, 3, 1)

        # Apply spectral gating
        weight = torch.view_as_complex(self.complex_weight)
        # print(x_fft.shape)
        # print(weight.shape)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')
        print("eeee",x.shape)

        return x.permute(0,3,2,1)

models = SpectralGatingNetwork(768,16,16)
x_input = torch.ones([2,768,16,16])
x_output = models(x_input)
print(x_output.shape)
