import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

sys.path.append(os.path.dirname('.'))

from benchmark.slimming.vgg import vgg16_bn

parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./apoz_prune_model.pth',
                    help='Path to model save')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

# train/valid dataset
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'),
                                   transform=val_transform)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True)

model = vgg16_bn(pretrained=False).to(args.device)
cudnn.benchmark = True

total = 0
# 64 x 2, 128 x 2, 256 x 3, 512 x 6, 4096 x 2
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
