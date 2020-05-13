import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim, utils
from torchvision import datasets, transforms
from lib.models.imagenet import fvgg16_bn
from lib.helper import ClassifyTrainer
from lib.utils import get_logger, save_pkl
from PIL import ImageFile

parser = argparse.ArgumentParser(description='Imagenet')
parser.add_argument('--model_name', type=str, default='vgg16_bn')
parser.add_argument('--datasets', type=str, default='imagenet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder')
parser.add_argument('--edge_filter_type', type=str, default='conv')
parser.add_argument('--texture_filter_type', type=str, default='normal')
parser.add_argument('--object_filter_type', type=str, default='normal')
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--log_path', type=str, default='./imagenet.log')
parser.set_defaults(feature=True)
args = parser.parse_args()

logger = get_logger(args.log_path)

torch.manual_seed(20145170)
torch.cuda.manual_seed(20145170)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# dataset / dataloader
train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'),
                                     transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)

test_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'),
                                    transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=8,
                                          pin_memory=True)

filter_types = [args.edge_filter_type,
                args.texture_filter_type,
                args.object_filter_type]

# model
if args.model_name == 'vgg16_bn':
    model = fvgg16_bn(filter_types=filter_types).to(args.device)

cudnn.benchmark = True

logger.info(f'MODEL : {args.model_name} \n'
            f'NUM Filter : {args.num_filters} \n'
            f'EDGE Filter : {args.edge_filter_type} \n'
            f'TEXTURE Filter : {args.texture_filter_type} \n'
            f'OBJECT Filter : {args.object_filter_type} \n'
            f'Learning Rate : {args.lr}')

# cost
criterion = nn.CrossEntropyLoss().to(args.device)

train_iter = len(train_loader)
test_iter = len(test_loader)

# optimizer/scheduler
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[50, 100, 150],
                                           gamma=0.1)

trainer = ClassifyTrainer(model,
                          criterion,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          optimizer=optimizer,
                          scheduler=None)

best_test_acc = 0
train_acc_log = []
train_loss_log = []
test_acc_log = []
test_loss_log = []

name = f'{args.datasets}_' \
       f'{args.model_name}_' \
       f'{args.num_filters}_' \
       f'{args.edge_filter_type}_' \
       f'{args.texture_filter_type}_' \
       f'{args.object_filter_type}'

# train
for e in range(args.epoch):
    scheduler.step()

    train_loss, train_acc = trainer.train()
    test_loss, test_acc = trainer.test()

    train_acc = train_acc / args.batch_size
    test_acc = test_acc / args.batch_size

    train_acc_log.append(train_acc)
    train_loss_log.append(train_loss)
    test_acc_log.append(test_acc)
    test_loss_log.append(test_loss)

    if test_acc > best_test_acc:
        trainer.save(f'{args.save_path}/{name}_model.pth')

        best_test_acc = test_acc

    logger.info(f"Epoch [ {args.epoch} / {e} ] \n"
                f" + TRAIN [Loss / Acc] : [ {train_loss} / {train_acc} ] \n"
                f" + TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")

save_pkl(train_acc_log, f'./pkl/{name}_train_acc_log')
save_pkl(train_loss_log, f'./pkl/{name}_train_loss_log')
save_pkl(test_acc_log, f'./pkl/{name}_test_acc_log')
save_pkl(test_loss_log, f'./pkl/{name}_test_loss_log')