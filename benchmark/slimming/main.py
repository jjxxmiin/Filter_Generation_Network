import os
import sys
import time
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import ImageFile
from torchvision import datasets
from torchvision import transforms

sys.path.append(os.path.dirname('.'))

from benchmark.slimming.vgg import vgg16_bn
from benchmark.helper import get_logger, accuracy, valid, AverageMeter


def updateBN(model, sparsity):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))


def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)

# setting
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Network Slimming')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./checkpoint/slimming_model.pth.tar',
                    help='Path to model save')
parser.add_argument('--epoch', type=int, default=90)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--sparsity', '-s', type=float, default=0.00001)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

logger = get_logger('./slimming.log')

# train/valid dataset
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_path = os.path.join(args.data_path, 'train')
valid_path = os.path.join(args.data_path, 'val')

train_dataset = datasets.ImageFolder(train_path,
                                     transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True)

valid_dataset = datasets.ImageFolder(valid_path,
                                     transform=valid_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           pin_memory=True)

# model
model = vgg16_bn(pretrained=False)
nn.DataParallel(model).to(args.device)
cudnn.benchmark = True

# loss
criterion = nn.CrossEntropyLoss().to(args.device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# train / valid
sparsity = args.sparsity
best_top1 = 0

for e in range(args.epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    train_iter = len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()

        if sparsity != 0:
            updateBN(model, sparsity)

        BN_grad_zero(model)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'Epoch: [{e}/{args.epoch}] \n'
                    f'Iter: [{i}/{train_iter}] \n'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \n'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f}) \n'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f}) \n'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \n')

    # valid
    top1, top5 = valid(model,
                       valid_loader,
                       criterion)

    logger.info(f"top1 : {top1} / top5 : {top5}")

    # save
    if top1 > best_top1:
        best_top1 = top1

        torch.save({'state_dict': model.state_dict()},
                   args.save_path)

