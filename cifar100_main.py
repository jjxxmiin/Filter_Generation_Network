import os
import argparse
import torch
from torch import nn, optim, utils
from torchvision import datasets, transforms
from lib.models.cifar100 import fvgg16_bn, fresnet18
from lib.models.module import get_filter
from lib.helper import ClassifyTrainer
from lib.utils import get_logger

parser = argparse.ArgumentParser(description='cifar100')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--datasets', type=str, default='cifar100')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=350)
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--edge_filter_type', '-e', type=str, default='conv')
parser.add_argument('--texture_filter_type', '-t', type=str, default='normal')
parser.add_argument('--object_filter_type', '-o', type=str, default='normal')
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--log_path', type=str, default='./cifar100.log')
parser.set_defaults(feature=True)
args = parser.parse_args()

logger = get_logger(args.log_path)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

torch.manual_seed(20145170)
torch.cuda.manual_seed(20145170)

# augmentation
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(size=(32, 32), padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

# dataset / dataloader
train_dataset = datasets.CIFAR100(root='../data',
                                  train=True,
                                  transform=train_transformer,
                                  download=True)

train_loader = utils.data.DataLoader(train_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True)

test_dataset = datasets.CIFAR100(root='../data',
                                 train=False,
                                 transform=test_transformer,
                                 download=True)

test_loader = utils.data.DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

first_filters = get_filter(args.edge_filter_type, num_filters=args.num_filters)
middle_filters = get_filter(args.texture_filter_type, num_filters=args.num_filters)
last_filters = get_filter(args.object_filter_type, num_filters=args.num_filters)

filters = [first_filters, middle_filters, last_filters]

# model
if args.model_name == 'vgg16':
    model = fvgg16_bn(filters=filters).to(args.device)

elif args.model_name == 'resnet18':
    model = fresnet18(filters=filters).to(args.device)

logger.info(f'MODEL : {args.model_name} \n'
            f'NUM Filter : {args.num_filters} \n'
            f'EDGE Filter : {args.edge_filter_type} \n'
            f'TEXTURE Filter : {args.texture_filter_type} \n'
            f'OBJECT Filter : {args.object_filter_type} \n'
            f'Learning Rate : {args.lr}')

name = f'{args.datasets}_' \
       f'{args.model_name}_' \
       f'{args.num_filters}_' \
       f'{args.edge_filter_type}_' \
       f'{args.texture_filter_type}_' \
       f'{args.object_filter_type}_'

# cost
criterion = nn.CrossEntropyLoss().to(args.device)

train_iter = len(train_loader)
test_iter = len(test_loader)

# optimizer/scheduler
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[150, 250],
                                           gamma=0.1)

trainer = ClassifyTrainer(model,
                          criterion,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          optimizer=optimizer,
                          scheduler=None)

best_test_acc = 0

# train
for e in range(args.epoch):
    scheduler.step()

    train_loss, train_acc = trainer.train()
    test_loss, test_acc, _ = trainer.test()

    train_acc = train_acc / args.batch_size
    test_acc = test_acc / args.batch_size

    if test_acc > best_test_acc:
        print("MODEL SAVED")
        trainer.save(f'{args.save_path}/{name}_model.pth')
        best_test_acc = test_acc

    logger.info(f"Epoch [ {args.epoch} / {e} ] \n"
                f" + TRAIN [Loss / Acc] : [ {train_loss} / {train_acc} ] \n"
                f" + TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")