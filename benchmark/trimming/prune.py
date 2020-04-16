import os
import sys
import torch
import argparse
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

sys.path.append(os.path.dirname('.'))

from benchmark.trimming.vgg import vgg16
from benchmark.trimming.apoz import APoZ
from benchmark.helper import save_pkl

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

model = vgg16(pretrained=True).to(args.device)
criterion = nn.CrossEntropyLoss().cuda()
apoz = APoZ(model).get_apoz(val_loader, criterion)

save_pkl(apoz, './vgg_apoz.pkl')
