import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.module import GFLayer, get_filter


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, filters, t=6, class_num=100):
        super().__init__()

        if filters is None:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * t, 1),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * t, 1),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                GFLayer(in_channels * t, in_channels * t, filters=filters, stride=stride, padding=1, groups=in_channels * t),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):

    def __init__(self, filter_types, num_filters, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        edge_filters = get_filter(filter_types[0], num_filters)
        texture_filters = get_filter(filter_types[1], num_filters)
        object_filters = get_filter(filter_types[2], num_filters)

        self.stage1 = LinearBottleNeck(32, 16, stride=1, filters=edge_filters, t=1)
        self.stage2 = self._make_stage(2, 16, 24, stride=2, filters=texture_filters, t=6)
        self.stage3 = self._make_stage(3, 24, 32, stride=2, filters=texture_filters, t=6)
        self.stage4 = self._make_stage(4, 32, 64, stride=2, filters=texture_filters, t=6)
        self.stage5 = self._make_stage(3, 64, 96, stride=1, filters=texture_filters, t=6)
        self.stage6 = self._make_stage(3, 96, 160, stride=1, filters=texture_filters, t=6)
        self.stage7 = LinearBottleNeck(160, 320, stride=1, filters=object_filters, t=6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, filters, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, filters, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, filters, t))
            repeat -= 1

        return nn.Sequential(*layers)


def mobilenetv2(filter_types, num_filters=3):
    return MobileNetV2(filter_types, num_filters)