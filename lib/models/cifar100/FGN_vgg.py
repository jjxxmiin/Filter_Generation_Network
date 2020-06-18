import torch
import torch.nn as nn
from lib.models.module import GFLayer, get_filter

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class FVGG(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, filters, batch_norm=False):
    first = [0, 1, 3, 4]
    middle = [6, 7, 8, 10, 11, 12]
    last = [14, 15, 16]

    edge_filters = filters[0]
    texture_filters = filters[1]
    object_filters = filters[2]

    layers = []

    input_channel = 3
    for i, l in enumerate(cfg):
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        if i in first:
            if edge_filters is None:
                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            else:
                layers += [GFLayer(input_channel, l, filters=edge_filters, stride=1, padding=1)]

        if i in middle:
            if texture_filters is None:
                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            else:
                layers += [GFLayer(input_channel, l, filters=texture_filters, stride=1, padding=1)]

        if i in last:
            if object_filters is None:
                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            else:
                layers += [GFLayer(input_channel, l, filters=object_filters, stride=1, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def fvgg16_bn(filters):
    return FVGG(make_layers(cfg['D'], filters, batch_norm=True))