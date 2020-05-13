import torch
import torch.nn as nn
from lib.models.module import GFLayer, get_filter

# torch.manual_seed(20145170)
# torch.cuda.manual_seed(20145170)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class FVGG(nn.Module):
    def __init__(self,
                 vgg_name,
                 num_filters,
                 filter_types,
                 stride=1):

        super(FVGG, self).__init__()

        # [0, 2, 4, 5, 7, 8, 10, 11]
        # [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
        self.first = [0, 1]
        self.middle = [3, 4, 6, 7, 8, 10, 11, 12]
        self.last = [14, 15, 16]

        self.stride = stride

        self.edge_filters = get_filter(filter_types[0], num_filters)
        self.texture_filters = get_filter(filter_types[1], num_filters)
        self.object_filters = get_filter(filter_types[2], num_filters)

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        i = 0

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i in self.first:
                    if self.edge_filters is None:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=self.stride, padding=1),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [GFLayer(in_channels, x, self.edge_filters, self.stride),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                elif i in self.middle:
                    if self.texture_filters is None:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=self.stride, padding=1),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [GFLayer(in_channels, x, self.texture_filters, self.stride),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                elif i in self.last:
                    if self.object_filters is None:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=self.stride, padding=1),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [GFLayer(in_channels, x, self.object_filters, self.stride),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]

                in_channels = x

            i += 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    filters = torch.autograd.Variable(torch.ones(3, 3, 3), requires_grad=True)

    model = FVGG('VGG16', filters)

    for p in model.parameters():
        print(p.shape)

    y = model(torch.randn(1, 3, 32, 32))
    print(y.size())