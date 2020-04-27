import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname('.'))

from benchmark.slimming.vgg import vgg16_bn

parser = argparse.ArgumentParser(description='Network Slimming')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./checkpoint/slimming_prune_model.pth.tar',
                    help='Path to model save')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--percent', type=float, default=0.1)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

model = vgg16_bn(pretrained=False).to(args.device)
cudnn.benchmark = True

# 64 x 2, 128 x 2, 256 x 3, 512 x 6, 4096 x 2
total = 0

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        total += m.weight.data.shape[0]

# batch norm weight save
# y = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma(weight) + alpha(bias)
bn = torch.zeros(total)
index = 0

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []

for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print(f"layer index: {k} \t total channel: {mask.shape[0]} \t remaining channel: {int(torch.sum(mask))}")

    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

# torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
# pruned_ratio = pruned/total
# print('Pre-processing Successful!')
