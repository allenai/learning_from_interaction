import torch
from torch import nn


class UpConv(nn.Module):
    """
    This module upconvolutes by a factor of 2, and also has a lateral connection typical of UNets.
    """
    def __init__(self, din, d_horizontal, dout):
        super(UpConv, self).__init__()
        self.upconv = nn.ConvTranspose2d(din, dout, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(d_horizontal + dout, dout, kernel_size=3, stride=1, padding=0),
                                  nn.BatchNorm2d(dout),
                                  nn.ReLU())

    def forward(self, x, y):
        x = self.upconv(x)
        x, y = self.crop(x, y)
        return self.conv(torch.cat([x, y], dim=1))

    @staticmethod
    def crop(x, y):
        width_x, width_y = x.shape[-1], y.shape[-1]
        if width_x == width_y: return x, y
        if width_x > width_y:
            low = (width_x - width_y) // 2
            high = width_x - width_y - low
            return x[:, :, low:-high, low:-high], y
        low = (width_y - width_x) // 2
        high = width_y - width_x - low
        return x, y[:, :, low:-high, low:-high]


class ResBlock(nn.Module):
    """
    This downconvolutes by a factor of 2. It is more light-weight than the original ResBlock.
    """
    def __init__(self, din, first_kernel_size=1, small=0):
        super(ResBlock, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(din, 2 * din, kernel_size=first_kernel_size, padding=(first_kernel_size - 1) // 2, stride=1,
                      bias=False, groups=din // small if small > 0 else 1),
            nn.BatchNorm2d(2 * din),
            nn.ReLU(),
            nn.Conv2d(2 * din, din, kernel_size=1, padding=0, stride=1,
                      bias=False, groups=1),
            nn.BatchNorm2d(din),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(din, 2 * din, kernel_size=3, padding=1, stride=2,
                      bias=False, groups=din // small if small > 0 else 1),
            nn.BatchNorm2d(2 * din),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.first_conv(x)
        return self.second_conv(x + y)
