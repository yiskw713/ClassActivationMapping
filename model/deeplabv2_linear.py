#!/usr/bin/env python
# coding: utf-8
#
# reference
#
##########################################################
#
# Cited from https://github.com/kazuto1011/deeplab-pytorch
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19
#
##########################################################

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


""" ResNet """

_BATCH_NORM = nn.BatchNorm2d

class _ConvBnReLU(nn.Sequential):
    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """Bottleneck Unit"""

    def __init__(self, in_ch, mid_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """Residual blocks"""

    def __init__(
        self, n_layers, in_ch, mid_ch, out_ch, stride, dilation, multi_grids=None
    ):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(
                multi_grids
            ), "{} values expected, but got: mg={}".format(n_layers, multi_grids)

        self.add_module(
            "block1",
            _Bottleneck(in_ch, mid_ch, out_ch, stride, dilation * multi_grids[0], True),
        )
        for i, rate in zip(range(2, n_layers + 1), multi_grids[1:]):
            self.add_module(
                "block" + str(i),
                _Bottleneck(out_ch, mid_ch, out_ch, 1, dilation * rate, False),
            )


class _Stem(nn.Sequential):
    """
    The 1st Residual Layer
    """

    def __init__(self):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, 64, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


""" DeepLabV2 """

class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(zip(rates)):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2_linear(nn.Module):
    """DeepLab v2 (OS=8)"""

    def __init__(self, obj_classes, aff_classes, n_blocks, atrous_rates):
        super().__init__()

        self.layer1 = _Stem()
        self.layer2 = _ResLayer(n_blocks[0], 64, 64, 256, 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4)
        self.aspp = _ASPP(2048, 32, atrous_rates)        

        self.conv_obj = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_aff = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_obj = nn.Linear(32, obj_classes, bias=False)
        self.fc_aff = nn.Linear(32, aff_classes, bias=False)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.aspp(x)

        y_obj = F.relu(self.conv_obj(x))
        y_obj = self.gap(y_obj)
        y_aff = F.relu(self.conv_aff(x))
        y_aff = self.gap(y_aff)

        y_obj = y_obj.view(x.shape[0], -1)
        y_aff = y_aff.view(x.shape[0], -1)

        y_obj = self.fc_obj(y_obj)
        y_aff = self.fc_aff(y_aff)

        return [y_obj, y_aff]

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DeepLabV2_linear_max(nn.Module):
    """DeepLab v2 (OS=8)"""

    def __init__(self, obj_classes, aff_classes, n_blocks, atrous_rates):
        super().__init__()

        self.layer1 = _Stem()
        self.layer2 = _ResLayer(n_blocks[0], 64, 64, 256, 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4)
        self.aspp = _ASPP(2048, 32, atrous_rates)        

        self.conv_obj = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_aff = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.fc_obj = nn.Linear(32, obj_classes, bias=False)
        self.fc_aff = nn.Linear(32, aff_classes, bias=False)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.aspp(x)

        y_obj = F.relu(self.conv_obj(x))
        y_obj = self.gmp(y_obj)
        y_aff = F.relu(self.conv_aff(x))
        y_aff = self.gmp(y_aff)

        y_obj = y_obj.view(x.shape[0], -1)
        y_aff = y_aff.view(x.shape[0], -1)

        y_obj = self.fc_obj(y_obj)
        y_aff = self.fc_aff(y_aff)

        return [y_obj, y_aff]

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
