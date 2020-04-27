import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, transforms, datasets

sys.path.append(os.path.dirname('.'))

from benchmark.helper import valid, train

parser = argparse.ArgumentParser(description='Weight Pruning')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./checkpoint/weight_prune_model.pth.tar',
                    help='Path to model save')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--percent', type=float, default=0.1)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

model = models.vgg16(pretrained=True).to(args.device)
cudnn.benchmark = True

# train/valid dataset
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_path = os.path.join(args.data_path, 'train')
valid_path = os.path.join(args.data_path, 'val')

valid_dataset = datasets.ImageFolder(valid_path,
                                     transform=valid_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           pin_memory=True)


criterion = nn.CrossEntropyLoss().to(args.device)

b_top1, b_top5 = valid(model,
                       valid_loader,
                       criterion)

total = 0

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        total += m.weight.data.numel()

conv_weights = torch.zeros(total).cuda()
index = 0

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        size = m.weight.data.numel()
        conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
        index += size

y, i = torch.sort(conv_weights)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
print(f'Pruning Threshold : {thre}')
zero_flag = False
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.numel() - torch.sum(mask)
        m.weight.data.mul_(mask)
        if int(torch.sum(mask)) == 0:
            zero_flag = True
        print(f'layer index: {k} \t total params: {mask.numel()} \t remaining params: {int(torch.sum(mask))}')

a_top1, a_top5 = valid(model,
                       valid_loader,
                       criterion)

print(f'Total conv params: {total}, Pruned conv params: {pruned}, Pruned ratio: {pruned / total}')
print(f"Befor Acc@1 : {b_top1} \t Acc@5 : {b_top5}")
print(f"After Acc@1 : {a_top1} \t Acc@5 : {a_top5}")

