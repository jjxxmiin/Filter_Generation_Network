import torch
import torch.nn as nn
from torch.utils import model_zoo
from lib.models.module import GFLayer, get_filter

__all__ = [
    'FVGG', 'fvgg11', 'fvgg11_bn', 'fvgg16', 'fvgg16_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class FVGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(FVGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, f_cfg, filter_types, num_filters=3, stride=1, batch_norm=False):
    edge_filters = get_filter(filter_types[0], num_filters=num_filters)
    pattern_filters = get_filter(filter_types[1], num_filters=num_filters)
    object_filters = get_filter(filter_types[2], num_filters=num_filters)

    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i in f_cfg[0]:
                if edge_filters is None:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=stride, padding=1)
                else:
                    conv2d = GFLayer(in_channels, v, edge_filters, stride)
            elif i in f_cfg[1]:
                if pattern_filters is None:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=stride, padding=1)
                else:
                    conv2d = GFLayer(in_channels, v, pattern_filters, stride)
            elif i in f_cfg[2]:
                if object_filters is None:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=stride, padding=1)
                else:
                    conv2d = GFLayer(in_channels, v, object_filters, stride)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

filter_cfgs = {
    'A': [[0, 2], [4, 5, 7, 8], [10, 11]],
    'D': [[0, 1, 3, 4], [6, 7, 8, 10, 11, 12], [14, 15, 16]],
    'D2': [[0, 1], [3, 4, 6, 7, 8, 10, 11, 12], [14, 15, 16]]
}


def _vgg(cfg, filter_types, batch_norm, **kwargs):
    model = FVGG(make_layers(cfg=cfgs[cfg],
                             f_cfg=filter_cfgs[cfg],
                             filter_types=filter_types,
                             batch_norm=batch_norm), **kwargs)

    return model


def fvgg11(filter_types, **kwargs):
    return _vgg('A', filter_types, False, **kwargs)


def fvgg11_bn(filter_types, **kwargs):
    return _vgg('A', filter_types, True, **kwargs)


def fvgg16(filter_types, **kwargs):
    return _vgg('D', filter_types, False, **kwargs)


def fvgg16_bn(filter_types, **kwargs):
    return _vgg('D', filter_types, True, **kwargs)
