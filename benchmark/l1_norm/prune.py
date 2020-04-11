import os
import sys
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname('.'))

from benchmark.l1_norm.resnet import resnet34
from utils.model_tools import *

parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./prune_model.pth',
                    help='Path to model save')
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
parser.add_argument('-v', type=str, default='A',
                    help='select [A / B]')

args = parser.parse_args()

skip = [2, 8, 14, 16, 26, 28, 30, 32]

prune_prob = {
    'A': [0.3, 0.3, 0.3, 0.0],
    'B': [0.5, 0.6, 0.4, 0.0],
}

model = resnet34(pretrained=True).to(args.device)
# model = torch.nn.DataParallel(model)
cudnn.benchmark = True

print("===== Origin =====")
print_model_param_nums(model)
print_model_param_flops(model)

layer_id = 1
filter_cfg = []
filter_mask = []

# model structure modify
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # 1 x 1 kernel size
        if m.kernel_size == (1, 1):
            continue

        out_channels = m.weight.data.shape[0]

        if layer_id in skip:
            filter_mask.append(torch.ones(out_channels))
            filter_cfg.append(out_channels)
            layer_id += 1

        elif layer_id % 2 == 0:
            # 64
            if layer_id <= 6:
                stage = 0
            # 128
            elif layer_id <= 14:
                stage = 1
            # 256
            elif layer_id <= 26:
                stage = 2
            # 512
            else:
                stage = 3

            prune_prob_stage = prune_prob[args.v][stage]
            weight_copy = abs(m.weight.data).clone().cpu().numpy()

            l1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            threshold = int(out_channels * (1 - prune_prob_stage))
            sorted_arg = np.argsort(l1_norm)
            sorted_arg_rev = sorted_arg[::-1][:threshold]

            mask = torch.zeros(out_channels)
            mask[sorted_arg_rev.tolist()] = 1

            filter_mask.append(mask)
            filter_cfg.append(threshold)

            layer_id += 1
        else:
            layer_id += 1

prune_model = resnet34(cfg=filter_cfg).to(args.device)
# prune_model = nn.DataParallel(prune_model)
print("===== Prune =====")
print_model_param_nums(prune_model)
print_model_param_flops(prune_model)

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1

print(filter_mask)

# model weight move
for [m0, m1] in zip(model.modules(), prune_model.modules()):
    if isinstance(m0, nn.Conv2d) and \
            isinstance(m1, nn.Conv2d):
        if m0.kernel_size == (1, 1):
            m1.weight.data = m0.weight.data.clone()

            continue

        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1

            continue

        if conv_count % 2 == 0:
            mask = filter_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))

            if idx.size == 1:
                idx = np.resize(idx, (1,))
            print(idx.shape)
            print(m0.weight.data.shape)
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1

            continue

        if conv_count % 2 == 1:
            mask = filter_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))

            if idx.size == 1:
                idx = np.resize(idx, (1,))

            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1

            continue

    elif isinstance(m0, nn.BatchNorm2d) and\
            isinstance(m1, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            mask = filter_mask[layer_id_in_cfg - 1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))

            if idx.size == 1:
                idx = np.resize(idx, (1,))

            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue

        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': filter_cfg,
            'state_dict': prune_model.state_dict()},
             args.save_path + '.tar')

