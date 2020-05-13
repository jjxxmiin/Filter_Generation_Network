import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO


class GFLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, filters):
        super(GFLayer, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.filters = filters
        self.num_filters = self.filters.size(0)
        self.weights = torch.nn.Parameter(torch.Tensor(out_ch, in_ch, self.num_filters))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_ch * self.num_filters * self.num_filters
        stdv = 1. / math.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        f = self.filters.view(1, 1, self.filters.size(0), self.filters.size(1), self.filters.size(2)) * \
                 self.weights.view(self.out_ch, self.in_ch, self.num_filters, 1, 1).repeat(1, 1, 1, 3, 3)
        f = f.sum(2)
        f = F.leaky_relu(f)
        output = F.conv2d(x, f, stride=1, padding=1)

        return output


class FGN(nn.Module):
    def __init__(self, num_filters=3):
        super(FGN, self).__init__()
        basis_filter = torch.autograd.Variable(torch.rand(num_filters, 3, 3)).to('cuda')

        self.gf1 = GFLayer(3, 32, basis_filter)
        self.gf2 = GFLayer(32, 32, basis_filter)

        self.classifier = nn.Sequential(nn.Linear(1568, 512),
                                        nn.Linear(512, 10))

    def forward(self, x):
        x = self.gf1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.gf2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    img = torch.autograd.Variable(torch.ones(1, 3, 32, 32), requires_grad=True)
    filters = torch.autograd.Variable(torch.ones(3, 3, 3), requires_grad=True)

    model = FGN(filters)

    for i in model.parameters():
        print(i.shape)