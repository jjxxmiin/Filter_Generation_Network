import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import random

from src.models.vgg import VGG
from src.loader import Sub_Tiny_ImageNet
from itertools import combinations

sys.path.append('.')

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'classify',
    'model': 'VGG19',
    'dataset': 'Tiny_ImageNet',
    'classes': 3,
    'lr': 0.01,
    'epochs': 100,
    'batch_size': 32,
    'input_shape': (3, 224, 224),
    'data_path': './datasets/tiny_imagenet'
}

type = 'imagenet_224_2'

if not os.path.exists(f'./models/{type}'):
    os.mkdir(f'./models/{type}')

# augmentation
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize(size=(configs['input_shape'][1],
                                                                configs['input_shape'][2])),
                                        transforms.RandomCrop(size=(configs['input_shape'][1],
                                                                    configs['input_shape'][2]), padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transformer = transforms.Compose([transforms.Resize(size=(224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# datasets/loader
train_path = os.path.join(configs['data_path'], 'train')

class_name = list(combinations(os.listdir(train_path), 3))

for subset in random.sample(class_name, 5):
    train_datasets = Sub_Tiny_ImageNet(configs['data_path'],
                                       subset,
                                       dtype='train',
                                       transformer=train_transformer)

    test_datasets = Sub_Tiny_ImageNet(configs['data_path'],
                                      subset,
                                      dtype='val',
                                      transformer=test_transformer)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=configs['batch_size'],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=configs['batch_size'],
                                              shuffle=True)

    print(subset)

    # model
    model = VGG(configs['model'], output=configs['classes']).to(device)

    # cost
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer/scheduler
    optimizer = optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                               milestones=[50],
                                               gamma=0.1)

    best_valid_acc = 0
    train_iter = len(train_loader)
    test_iter = len(test_loader)

    # train
    for epoch in range(configs['epochs']):
        train_loss = 0
        valid_loss = 0

        n_train_correct = 0
        n_valid_correct = 0

        scheduler.step()

        for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
            model.train()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # forward
            pred = model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            n_train_correct += (predicted == labels).sum().item()
            # loss
            loss = criterion(pred, labels)
            train_loss += loss.item()
            # backward
            loss.backward()
            # weight update
            optimizer.step()

        train_acc = n_train_correct / (train_iter * configs['batch_size'])
        train_loss = train_loss / train_iter

        model.eval()
        for images, labels in test_loader:
            images, label = images.to(device), labels.to(device)

            pred = model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            n_valid_correct += (predicted == labels).sum().item()
            # loss
            loss = criterion(pred, labels)
            valid_loss += loss.item()

        valid_acc = n_valid_correct / (test_iter * configs['batch_size'])
        valid_loss = valid_loss / test_iter

        print(f"\nEpoch [ {configs['epochs']} / {epoch} ] "
              f"TRAIN [Acc / Loss] : [ {train_acc} / {train_loss} ]"
              f" TEST [Acc / Loss] : [ {valid_acc} / {valid_loss} ]")

        if valid_acc > best_valid_acc:
            print("model saved")
            torch.save(model.state_dict(), f"./models/{type}/{subset}.pth")
            best_valid_acc = valid_acc
