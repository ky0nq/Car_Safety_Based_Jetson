"""MiniFASNetV2 architecture used by the official pretrained weight.

Adapted from minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0).
Original project: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""
from __future__ import annotations

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, stride=(2, 2)):
        super().__init__()
        self.conv = ConvBlock(c1[0], c1[1])
        self.conv_dw = ConvBlock(c2[0], c2[1], kernel=(3, 3), stride=stride, padding=(1, 1), groups=c2[0])
        self.project = LinearBlock(c3[0], c3[1])
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.project(self.conv_dw(self.conv(x)))
        return shortcut + x if self.residual else x


class Residual(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__()
        self.model = nn.Sequential(
            *(DepthWise(c1[i], c2[i], c3[i], residual=True, stride=(1, 1)) for i in range(len(c1)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


KEEP_V2 = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13,
    64, 231, 231, 128, 231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77,
    128, 26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26,
    128, 512, 512,
]


class MiniFASNetV2(nn.Module):
    def __init__(self, embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3):
        super().__init__()
        k = KEEP_V2
        self.embedding_size = embedding_size
        self.conv1 = ConvBlock(3, k[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(k[0], k[1], kernel=(3, 3), padding=(1, 1), groups=k[1])
        self.conv_23 = DepthWise((k[1], k[2]), (k[2], k[3]), (k[3], k[4]))

        c1 = [(k[4], k[5]), (k[7], k[8]), (k[10], k[11]), (k[13], k[14])]
        c2 = [(k[5], k[6]), (k[8], k[9]), (k[11], k[12]), (k[14], k[15])]
        c3 = [(k[6], k[7]), (k[9], k[10]), (k[12], k[13]), (k[15], k[16])]
        self.conv_3 = Residual(c1, c2, c3)

        self.conv_34 = DepthWise((k[16], k[17]), (k[17], k[18]), (k[18], k[19]))

        c1 = [(k[19], k[20]), (k[22], k[23]), (k[25], k[26]), (k[28], k[29]), (k[31], k[32]), (k[34], k[35])]
        c2 = [(k[20], k[21]), (k[23], k[24]), (k[26], k[27]), (k[29], k[30]), (k[32], k[33]), (k[35], k[36])]
        c3 = [(k[21], k[22]), (k[24], k[25]), (k[27], k[28]), (k[30], k[31]), (k[33], k[34]), (k[36], k[37])]
        self.conv_4 = Residual(c1, c2, c3)

        self.conv_45 = DepthWise((k[37], k[38]), (k[38], k[39]), (k[39], k[40]))

        c1 = [(k[40], k[41]), (k[43], k[44])]
        c2 = [(k[41], k[42]), (k[44], k[45])]
        c3 = [(k[42], k[43]), (k[45], k[46])]
        self.conv_5 = Residual(c1, c2, c3)

        self.conv_6_sep = ConvBlock(k[46], k[47])
        self.conv_6_dw = LinearBlock(k[47], k[48], kernel=conv6_kernel, groups=k[48])
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        if self.embedding_size != 512:
            x = self.linear(x)
        return self.prob(self.drop(self.bn(x)))
