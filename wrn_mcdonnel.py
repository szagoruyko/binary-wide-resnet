from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.conv0 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        h = self.conv0(F.relu(self.bn0(x)))
        h = self.conv1(F.relu(self.bn1(h)))
        return x + h


class DownsampleBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.conv0 = nn.Conv2d(width // 2, width,
                               kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.conv1 = nn.Conv2d(width, width,
                               kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        h = self.conv0(F.relu(self.bn0(x)))
        h = self.conv1(F.relu(self.bn1(h)))
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN_McDonnel(nn.Module):
    """Implementation of modified Wide Residual Network.
    """

    def __init__(self, depth, width, num_classes):
        super().__init__()
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6

        self.conv0 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)

        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)

        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        self.conv_last = nn.Conv2d(widths[2], num_classes, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(num_classes)

    @staticmethod
    def _make_block(width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width)
            return Block(width)
        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i))
                                         for i in range(n)))

    def forward(self, x):
        h = self.conv0(x)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = F.relu(self.bn(h))
        h = self.conv_last(h)
        h = self.bn_last(h)
        return F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)
