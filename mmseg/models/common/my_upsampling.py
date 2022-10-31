import torch
import torch.nn as nn


class MyUpsampling(nn.Module):
    def __init__(self, c_in, scale_factor=2):
        super(MyUpsampling, self).__init__()
        self.conv = nn.Conv2d(c_in, 4 * c_in, kernel_size=3, padding=1, stride=1)
        self.pix_shf = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pix_shf(x)
        x = self.relu(x)
        return x