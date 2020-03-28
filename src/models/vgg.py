'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,  256, 'M', 512, 'M', 512, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, output=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, output)
        self.classifier = nn.Linear(2048, output)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


def load_model(path, type='VGG16', mode='eval', device='cuda'):
    model = VGG(type, output=3).to(device)
    model.load_state_dict(torch.load(path))

    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        AssertionError("MODE is only train and eval")

    return model


def get_layer_index(version='VGG16'):
    idx = []

    for i in cfg[version]:
        if i == 'M':
            pass
        else:
            idx.append(np.arange(i))

    return idx


def get_layer_info(model):
    info = []

    for i, m in model.features.named_children():
        if type(m) == nn.Conv2d:
            info.append(f'Conv {m.in_channels} -> {m.out_channels}')
        elif type(m) == nn.BatchNorm2d:
            info.append(f'Bn   {m.num_features}')

    return info
