import os
import sys
import argparse

sys.path.append(os.path.dirname('.'))

from benchmark.trimming.vgg import vgg16
from benchmark.helper import load_pkl

parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./apoz_prune_model.pth',
                    help='Path to model save')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

# trimming CONV 5-3, FC6, FC7

model = vgg16(pretrained=True).to(args.device)
apoz = load_pkl('./vgg_apoz.pkl')

for p in apoz:
    print(100 - p.mean() * 100)
