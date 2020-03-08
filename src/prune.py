import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from src.loader import Sub_Binary_CIFAR, Sub_CIFAR
from src.converter import *
from tqdm import tqdm


def prune(model, filters):
    conv_id = []
    bn_id = []

    for i, m in model.features.named_children():
        if type(m) == nn.Conv2d:
            conv_id.append(int(i))
        elif type(m) == nn.BatchNorm2d:
            bn_id.append(int(i))

    if len(conv_id) is not len(filters):
        AssertionError("Conv do not match")

    for i, (c_id, b_id) in enumerate(zip(conv_id, bn_id)):
        if i == 0:
            new_conv = cvt_first_conv2d(model.features[c_id], filters[i])
            new_bn = cvt_bn2d(model.features[b_id], filters[i])
        elif i == (len(filters) - 1):
            new_conv = cvt_last_conv2d(model.features[c_id], filters[i - 1])
            new_bn = cvt_last_bn2d(model.features[b_id])
        else:
            new_conv = cvt_middle_conv2d(model.features[c_id], filters[i - 1], filters[i])
            new_bn = cvt_bn2d(model.features[b_id], filters[i])

        model.features[c_id] = new_conv
        model.features[b_id] = new_bn

    return model


def to_binary(model, c):
    model.classifier = cvt_binary_linear(model.classifier, c)

    return model


def binary_train(model, configs, c, logger, device='cuda'):
    model.train()
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size=(32, 32), padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_datasets = Sub_Binary_CIFAR(root_path=configs['root_path'],
                                      class_names=['airplane', 'dog', 'horse'],
                                      c=c,
                                      dtype='train',
                                      transformer=transformer)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=configs['batch_size'],
                                               shuffle=True, )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=5e-4)

    train_iter = len(train_loader)

    train_loss = 0
    n_train_correct = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
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
        loss.backward(retain_graph=True)
        # weight update
        optimizer.step()

        train_acc = n_train_correct / (train_iter * configs['batch_size'])
        train_loss = train_loss / train_iter

    logger.info(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]")

    return model


def binary_test(model, configs, c, logger, device='cuda'):
    model.eval()
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_datasets = Sub_Binary_CIFAR(root_path=configs['root_path'],
                                     class_names=['airplane', 'dog', 'horse'],
                                     c=c,
                                     dtype='test',
                                     transformer=transformer)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=configs['batch_size'],
                                              shuffle=True, )

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_iter = len(test_loader)
    test_loss = 0
    n_test_correct = 0

    for i, (images, labels) in tqdm(enumerate(test_loader), total=test_iter):
        images, labels = images.to(device), labels.to(device)
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * configs['batch_size'])
        test_loss = test_loss / test_iter

    logger.info(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")


def train(model, configs, logger, device='cuda'):
    model.train()
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(size=(32, 32), padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_datasets = Sub_CIFAR(root_path=configs['root_path'],
                               class_names=['airplane', 'dog', 'horse'],
                               dtype='train',
                               transformer=transformer)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=configs['batch_size'],
                                               shuffle=True, )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=5e-4)

    train_iter = len(train_loader)

    train_loss = 0
    n_train_correct = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
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
        loss.backward(retain_graph=True)
        # weight update
        optimizer.step()

        train_acc = n_train_correct / (train_iter * configs['batch_size'])
        train_loss = train_loss / train_iter

    logger.info(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]" )

    return model


def test(model, configs, logger, device='cuda'):
    model.eval()
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_datasets = Sub_CIFAR(root_path=configs['root_path'],
                              class_names=['airplane', 'dog', 'horse'],
                              dtype='test',
                              transformer=transformer)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=configs['batch_size'],
                                              shuffle=True, )

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_iter = len(test_loader)
    test_loss = 0
    n_test_correct = 0

    for i, (images, labels) in tqdm(enumerate(test_loader), total=test_iter):
        images, labels = images.to(device), labels.to(device)
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * configs['batch_size'])
        test_loss = test_loss / test_iter

    logger.info(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")
