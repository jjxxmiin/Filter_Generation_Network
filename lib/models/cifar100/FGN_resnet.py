import torch
import torch.nn as nn
from lib.models.module import GFLayer, get_filter


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, filters, stride=1):
        super().__init__()

        if filters is None:
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self.residual_function = nn.Sequential(
                GFLayer(in_channels, out_channels, filters=filters, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                GFLayer(out_channels, out_channels * BasicBlock.expansion, filters=filters, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, filter_types, num_filters, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        edge_filters = get_filter(filter_types[0], num_filters)
        texture_filters = get_filter(filter_types[1], num_filters)
        object_filters = get_filter(filter_types[2], num_filters)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0],  filters=edge_filters, stride=1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], filters=texture_filters, stride=2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], filters=texture_filters, stride=2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], filters=object_filters, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, filters, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, filters, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def fresnet18(filter_types, num_filter=3):
    return ResNet(BasicBlock, filter_types, num_filter, [2, 2, 2, 2])


def fresnet34(filter_types, num_filter=3):
    return ResNet(BasicBlock, filter_types, num_filter, [3, 4, 6, 3])


def fresnet50(filter_types, num_filter=3):
    return ResNet(Bottleneck, filter_types, num_filter, [3, 4, 6, 3])


def fresnet101(filter_types, num_filter=3):
    return ResNet(Bottleneck, filter_types, num_filter, [3, 4, 23, 3])


def fresnet152(filter_types, num_filter=3):
    return ResNet(Bottleneck, filter_types, num_filter, [3, 8, 36, 3])
