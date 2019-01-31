import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    '''
    conv => batchnorm => relu
    '''

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out


class VGG16(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        # input_size = (256, 256)

        self.features = nn.Sequential(
            Conv(in_channel, 64),
            Conv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv(64, 128),
            Conv(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv(128, 256),
            Conv(256, 256),
            Conv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv(256, 512),
            Conv(512, 512),
            Conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv(512, 512),
            Conv(512, 512),
            Conv(512, 512),
            nn.AvgPool2d((16, 16))  # Global Average Pooling
        )

        self.classifier = nn.Linear(512, out_channel, bias=True)

    def forward(self, x):
        x = self.features(x)

        x = x.view(-1, 512)

        out = self.classifier(x)

        return out




