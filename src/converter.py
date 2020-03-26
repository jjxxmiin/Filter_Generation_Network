import torch
import numpy as np
import copy
import logging


def cvt_first_conv2d(conv, post, device='cuda'):
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                               out_channels=len(post),
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    ori_weights = conv.weight.data.cpu().numpy()
    new_weights = ori_weights[post, :]

    logging.info(f"first conv2d : {new_weights.shape}")

    new_conv.weight.data = torch.from_numpy(new_weights)
    new_conv.weight.data = new_conv.weight.data.cuda()

    return new_conv


def cvt_middle_conv2d(conv, pre, post, device='cuda'):
    new_conv = torch.nn.Conv2d(in_channels=len(pre),
                               out_channels=len(post),
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = old_weights[post, :]
    new_weights = new_weights[:, pre]

    logging.info(f"middle conv2d : {new_weights.shape}")

    new_conv.weight.data = torch.from_numpy(new_weights)
    new_conv.weight.data = new_conv.weight.data.cuda()

    return new_conv


def cvt_last_conv2d(conv, pre, device='cuda'):
    new_conv = torch.nn.Conv2d(in_channels=len(pre),
                               out_channels=conv.out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=(conv.bias is not None)).to(device)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = old_weights[:, pre]

    logging.info(f"last conv2d : {new_weights.shape}")

    new_conv.weight.data = torch.from_numpy(new_weights)
    new_conv.weight.data = new_conv.weight.data.cuda()

    return new_conv


def cvt_bn2d(bn, post):
    new_bn = copy.deepcopy(bn)
    new_bn.weight.data = new_bn.weight.data[post]
    new_bn.bias.data = new_bn.bias.data[post]
    new_bn.running_mean.data = new_bn.running_mean.data[post]
    new_bn.running_var.data = new_bn.running_var.data[post]

    return new_bn


def cvt_last_bn2d(bn):
    new_bn = copy.deepcopy(bn)

    new_bn.weight.data = new_bn.weight.data
    new_bn.bias.data = new_bn.bias.data
    new_bn.running_mean.data = new_bn.running_mean.data
    new_bn.running_var.data = new_bn.running_var.data

    return new_bn


def cvt_binary_sigmoid_linear(linear, cls, device='cuda'):
    new_linear = torch.nn.Linear(in_features=linear.in_features,
                                 out_features=1,
                                 bias=(linear.bias is not None)).to(device)

    new_weights = linear.weight.data.cpu().numpy()[cls]
    new_weights = np.expand_dims(new_weights, 0)
    new_linear.weight.data = torch.from_numpy(new_weights).cuda()

    new_bias = linear.bias.data.cpu().numpy()[cls]
    new_bias = np.expand_dims(new_bias, 0)
    new_linear.bias.data = torch.from_numpy(new_bias).cuda()

    return new_linear


def cvt_binary_linear(linear, cls, device='cuda'):
    new_linear = torch.nn.Linear(in_features=linear.in_features,
                                 out_features=2,
                                 bias=(linear.bias is not None)).to(device)

    cnt = 0
    false_weights = 0
    true_weights = None

    for i, w in enumerate(linear.weight.data.cpu().numpy()):
        if i == cls:
            true_weights = np.expand_dims(w, 0)
        else:
            cnt += 1
            false_weights = w

    false_weights = np.expand_dims(false_weights, 0)
    new_weights = np.concatenate((true_weights, false_weights), axis=0)
    new_linear.weight.data = torch.from_numpy(new_weights).cuda()

    new_bias = linear.bias.data.cpu().numpy()[cls]
    new_bias = np.expand_dims(new_bias, 0)
    new_linear.bias.data = torch.from_numpy(new_bias).cuda()

    return new_linear
