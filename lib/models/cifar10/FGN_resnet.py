import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.module import GFLayer, get_filter


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, filters, stride=1):
        super(BasicBlock, self).__init__()
        if filters is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = GFLayer(in_planes, planes, filters=filters, stride=stride, padding=1)
            self.conv2 = GFLayer(planes, planes, filters=filters, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, filter_types, num_filters, num_blocks, num_classes=10):

        super(ResNet, self).__init__()

        edge_filters = get_filter(filter_types[0], num_filters)
        texture_filters = get_filter(filter_types[1], num_filters)
        object_filters = get_filter(filter_types[2], num_filters)

        self.in_planes = 64

        if edge_filters is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = GFLayer(3, 64, filters=edge_filters, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], filters=texture_filters, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], filters=texture_filters, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], filters=texture_filters, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], filters=object_filters, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, filters, stride):
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, filters, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def fresnet18(filter_types, num_filters=3):
    return ResNet(BasicBlock, filter_types, num_filters, [2, 2, 2, 2])


def fresnet34(filter_types, num_filters=3):
    return ResNet(BasicBlock, filter_types, num_filters, [3, 4, 6, 3])


def fresnet50(filter_types, num_filters=3):
    return ResNet(Bottleneck, filter_types, num_filters, [3, 4, 6, 3])


def fresnet101(filter_types, num_filters=3):
    return ResNet(Bottleneck, filter_types, num_filters, [3, 4, 23, 3])


def fresnet152(filter_types, num_filters=3):
    return ResNet(Bottleneck, filter_types, num_filters, [3, 8, 36, 3])