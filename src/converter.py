import torch
import numpy as np


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

    print(f"first conv2d : {new_weights.shape}")

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

    print(f"middle conv2d : {new_weights.shape}")

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

    print(f"last conv2d : {new_weights.shape}")

    new_conv.weight.data = torch.from_numpy(new_weights)
    new_conv.weight.data = new_conv.weight.data.cuda()

    return new_conv


def cvt_bn2d(bn, post, device='cuda'):
    new_bn = torch.nn.BatchNorm2d(num_features=len(post),
                                  eps=bn.eps,
                                  momentum=bn.momentum,
                                  affine=bn.affine,
                                  track_running_stats=bn.track_running_stats).to(device)

    old_weights = bn.weight.data.cpu().numpy()
    new_weights = old_weights[post]

    new_bn.weight.data = torch.from_numpy(new_weights)
    new_bn.weight.data = new_bn.weight.data.cuda()

    return new_bn


def cvt_last_bn2d(bn, device='cuda'):
    new_bn = torch.nn.BatchNorm2d(num_features=bn.num_features,
                                  eps=bn.eps,
                                  momentum=bn.momentum,
                                  affine=bn.affine,
                                  track_running_stats=bn.track_running_stats).to(device)

    new_bn.weight.data = torch.from_numpy(bn.weight.data.cpu().numpy())
    new_bn.weight.data = new_bn.weight.data.cuda()

    return new_bn


def cvt_binary_linear(linear, cls, device='cuda'):
    new_linear = torch.nn.Linear(in_features=linear.in_features,
                                 out_features=2,
                                 bias=(linear.bias is not None)).to(device)

    false_weights = 0
    true_weights = None

    for i, w in enumerate(linear.weight.data.cpu().numpy()):
        if i == cls:
            true_weights = np.expand_dims(w, 0)
        else:
            false_weights = np.minimum(false_weights, w)

    false_weights = np.expand_dims(false_weights, 0)

    new_weights = np.concatenate((true_weights, false_weights), axis=0)
    new_linear.weight.data = torch.from_numpy(new_weights).cuda()

    new_bias = linear.bias.data.cpu().numpy()[cls]
    new_bias = np.expand_dims(new_bias, 0)
    new_linear.bias.data = torch.from_numpy(new_bias).cuda()

    return new_linear
