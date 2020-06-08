import torch.nn as nn
from lib.models.module import GFLayer, get_filter


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class FVGG(nn.Module):
    def __init__(self,
                 features,
                 num_class=10):

        super(FVGG, self).__init__()

        self.features = features
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def make_layers(cfg, filter_types, num_filters, bias, batch_norm=False):
    first = [0, 1]
    middle = [3, 4, 6, 7, 8, 10, 11, 12]
    last = [14, 15, 16]

    edge_filters = get_filter(filter_types[0], num_filters)
    texture_filters = get_filter(filter_types[1], num_filters)
    object_filters = get_filter(filter_types[2], num_filters)

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
                layers += [GFLayer(input_channel, l, filters=edge_filters, stride=1, padding=1, bias=bias)]

        if i in middle:
            if texture_filters is None:
                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            else:
                layers += [GFLayer(input_channel, l, filters=texture_filters, stride=1, padding=1, bias=bias)]

        if i in last:
            if object_filters is None:
                layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            else:
                layers += [GFLayer(input_channel, l, filters=object_filters, stride=1, padding=1, bias=bias)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def fvgg16_bn(filter_types, num_filters=3, bias=True):
    return FVGG(make_layers(cfg['D'], filter_types, num_filters, bias=bias, batch_norm=True))