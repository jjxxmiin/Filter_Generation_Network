import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GFLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, filters, stride, padding=1, bias=True):
        super(GFLayer, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.weights = nn.Parameter(torch.Tensor(out_ch, in_ch, self.filters.size(0)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_ch * 3 * 3
        stdv = 1. / math.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        f = self.filters.view(1, 1, self.filters.size(0), self.filters.size(1), self.filters.size(2)) * \
                 self.weights.view(self.out_ch, self.in_ch, self.filters.size(0), 1, 1).repeat(1, 1, 1, 3, 3)
        f = f.sum(2)

        output = F.conv2d(x, f, stride=self.stride, padding=self.padding)

        return output


def get_filter(filter_type, num_filters, device='cuda'):
    if filter_type == 'uniform':
        # uniform distribution [r1, r2)
        r1 = -3
        r2 = 3
        filters = torch.autograd.Variable((r1 - r2) * torch.rand(num_filters, 3, 3) + r2).to(device)

    elif filter_type == 'normal':
        # normal distribution mean : 0 variance : 1
        filters = torch.autograd.Variable(torch.randn(num_filters, 3, 3)).to(device)

    elif filter_type == 'exp':
        filters = torch.autograd.Variable(torch.from_numpy(np.random.exponential(size=(num_filters, 3, 3))).float()).to(device)

    elif filter_type == 'sobel':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                       [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                       [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                                       [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]])).to(device)

    elif filter_type == 'sobel_roberts':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                       [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                       [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
                                                       [[-1, 0, 0], [0, 1, 0], [0, 0, 0]]])).to(device)

    elif filter_type == 'sobel_raplacian':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                                        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]])).to(device)

    elif filter_type == 'line':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
                                                        [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
                                                        [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
                                                        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]])).to(device)

    elif filter_type == 'custom':
        filters = torch.autograd.Variable(torch.Tensor([[[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                                                        [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
                                                        [[1, 1, 0], [0, 0, 1], [0, 0, 0]],
                                                        [[0, 0, 0], [1, 0, 0], [1, 1, 0]]])).to(device)

    else:
        print('Conv Filter')
        filters = None

    return filters