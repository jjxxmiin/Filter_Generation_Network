import os
import torch
import argparse
from torch import nn, optim, utils
from torchvision import datasets, transforms
from lib.models.mnist.FGN import FGN
from lib.helper import ClassifyTrainer
from lib.utils import get_logger

torch.manual_seed(20145170)
torch.cuda.manual_seed(20145170)

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--num_filter', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--log_path', type=str, default='./mnist.log')

args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

logger = get_logger(args.log_path)

# augmentation
train_transformer = transforms.Compose([transforms.Grayscale(3),
                                        transforms.ToTensor()])

test_transformer = transforms.Compose([transforms.Grayscale(3),
                                       transforms.ToTensor()])

# dataset / dataloader
train_dataset = datasets.MNIST(root='../data',
                               train=True,
                               transform=train_transformer,
                               download=True)

train_loader = utils.data.DataLoader(train_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True)

test_dataset = datasets.MNIST(root='../data',
                              train=False,
                              transform=test_transformer,
                              download=True)

test_loader = utils.data.DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

model = FGN().to(args.device)

# cost
criterion = nn.CrossEntropyLoss().to(args.device)

if args.device == 'cuda':
    criterion = criterion.to(args.device)

train_iter = len(train_loader)
test_iter = len(test_loader)

# optimizer/scheduler
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

trainer = ClassifyTrainer(model,
                          criterion,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          optimizer=optimizer,
                          scheduler=None)

best_test_acc = 0

# train
for e in range(args.epoch):
    train_loss, train_acc = trainer.train()
    test_loss, test_acc, _ = trainer.test()

    train_acc = train_acc / args.batch_size
    test_acc = test_acc / args.batch_size

    if test_acc > best_test_acc:
        trainer.save(f'{args.save_path}/mnist.pth')
        best_test_acc = test_acc

    logger.info(f"Epoch [ {args.epoch} / {e} ] \n"
                f" + TRAIN [Loss / Acc] : [ {train_loss} / {train_acc} ] \n"
                f" + TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")