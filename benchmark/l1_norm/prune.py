'''https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/l1-norm-pruning/prune.py'''

import os
import sys
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname('.'))

from benchmark.l1_norm.resnet import resnet34
from benchmark.helper import test, accuracy
from utils.model_tools import *

parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./l1_prune_model.pth',
                    help='Path to model save')
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
parser.add_argument('-v', type=str, default='A',
                    help='select [A / B]')

args = parser.parse_args()

skip = {
    'A': [2, 8, 14, 16, 26, 28, 30, 32],
    'B': [2, 8, 14, 16, 26, 28, 30, 32],
}

prune_prob = {
    'A': [0.3, 0.3, 0.3, 0.0],
    'B': [0.5, 0.6, 0.4, 0.0],
}


model = resnet34(pretrained=True)
model = torch.nn.DataParallel(model).to(args.device)
cudnn.benchmark = True

print("===== Origin =====")
print_model_param_nums(model)
print_model_param_flops(model)

layer_id = 1
cfg = []
cfg_mask = []

# Search
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1,1):
            continue
        out_channels = m.weight.data.shape[0]
        if layer_id in skip[args.v]:
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            continue
        if layer_id % 2 == 0:
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
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1,2,3))
            num_keep = int(out_channels * (1 - prune_prob_stage))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue

        layer_id += 1

prune_model = resnet34(cfg=cfg)
prune_model = torch.nn.DataParallel(prune_model).to(args.device)

print("===== Prune =====")
print_model_param_nums(prune_model)
print_model_param_flops(prune_model)

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1

# Pruning
for [m0, m1] in zip(model.modules(), prune_model.modules()):
    if isinstance(m0, nn.Conv2d):
        if m0.kernel_size == (1, 1):
            # Cases for down-sampling convolution.
            m1.weight.data = m0.weight.data.clone()
            continue

        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue

        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue

        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue

    elif isinstance(m0, nn.BatchNorm2d):
        assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."

        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
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


torch.save({'cfg': cfg,
            'state_dict': prune_model.state_dict()},
             args.save_path + '.tar')

acc_top1, acc_top5 = test(model, args.data_path)
prune_acc_top1, prune_acc_top5 = test(prune_model, args.data_path)

num_ori_parameters = sum([param.nelement() for param in model.parameters()])
num_pru_parameters = sum([param.nelement() for param in prune_model.parameters()])

print(f"Before Pruning \n"
      f"Acc@1: {acc_top1} \n"
      f"Acc@5: {acc_top5} \n"
      f"Param: {num_ori_parameters} \n \n"
      f"After Pruning \n"
      f"Acc@1: {prune_acc_top1} \n"
      f"Acc@5: {prune_acc_top5} \n"
      f"Param: {num_pru_parameters}")
