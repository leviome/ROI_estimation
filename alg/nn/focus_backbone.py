import torch.nn.functional as F
import torch.nn as nn

layer_configs = [
    # Unit1 (2)
    (32, 3, 1),
    (64, 3, 1),
    # Unit2 (3)
    (128, 3, 0),
    (64, 1, 0),
    (128, 3, 1),
    # Unit3 (3)
    (256, 3, 1),
    (128, 1, 0),
    (256, 3, 0),
    # Unit4 (5)
    (512, 3, 0),
    (256, 1, 0),
    (512, 3, 0),
    (256, 1, 0),
    (512, 3, 0),
    # Unit5 (5)
    (1024, 3, 2),
    (512, 1, 0),
    (1024, 3, 0),
    (512, 1, 0),
    (1024, 3, 0),
    (1, 3, 0)
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels_num, out_channels_num, kernel_size, pool, stride=1):
        super(ConvBlock, self).__init__()
        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels_num, out_channels_num, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels_num)
        self.act = nn.LeakyReLU(0.1)
        self.pool = pool

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.pool == 1:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        elif self.pool == 2:
            out = F.max_pool2d(out, kernel_size=(3, 5), stride=(3, 5))
        return out


class NNBackbone(nn.Module):
    def __init__(self):
        super(NNBackbone, self).__init__()
        self.layer_setting = layer_configs
        self.feature = self.make_layers(3)

    def make_layers(self, in_channels_num):
        layers = []
        input_temp = in_channels_num
        for output, kernel_size, pool in self.layer_setting:
            layers.append(ConvBlock(input_temp, output, kernel_size, pool))
            input_temp = output
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.feature(x)
        return output
