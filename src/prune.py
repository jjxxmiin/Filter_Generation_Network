import logging
import torch.nn as nn
import torch.optim as optim
from src.converter import *

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


def binary_train(model, loader, batch_size, lr, prog, device='cuda'):
    model.train()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_iter = len(loader)

    train_loss = 0
    n_train_correct = 0

    for i, (images, labels) in enumerate(loader):
        prog.setValue(100 / train_iter * (i + 1))

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

        train_acc = n_train_correct / (train_iter * batch_size)
        train_loss = train_loss / train_iter

    logging.info(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]")

    return model, train_acc


def binary_test(model, loader, batch_size, prog, device='cuda'):
    model.eval()

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_iter = len(loader)
    test_loss = 0
    n_test_correct = 0

    for i, (images, labels) in enumerate(loader):
        prog.setValue(100 / test_iter * (i + 1))

        images, labels = images.to(device), labels.to(device)
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * batch_size)
        test_loss = test_loss / test_iter

    logging.info(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")

    return test_acc


def train(model, loader, batch_size, lr, prog, device='cuda'):
    model.train()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_iter = len(loader)
    train_loss = 0
    n_train_correct = 0

    for i, (images, labels) in enumerate(loader):
        prog.setValue(100 / train_iter * (i + 1))

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

        train_acc = n_train_correct / (train_iter * batch_size)
        train_loss = train_loss / train_iter

    logging.info(f"[TRAIN Acc / {train_acc}] [TRAIN Loss / {train_loss}]" )

    return model, train_acc


def test(model, loader, batch_size, prog, device='cuda'):
    model.eval()

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_iter = len(loader)
    test_loss = 0
    n_test_correct = 0

    for i, (images, labels) in enumerate(loader):
        prog.setValue(100 / test_iter * (i + 1))

        images, labels = images.to(device), labels.to(device)
        # forward
        pred = model(images)
        # acc
        _, predicted = torch.max(pred, 1)

        n_test_correct += (predicted == labels).sum().item()
        # loss
        loss = criterion(pred, labels)
        test_loss += loss.item()

        test_acc = n_test_correct / (test_iter * batch_size)
        test_loss = test_loss / test_iter

    logging.info(f"[TEST Acc / {test_acc}] [TEST Loss / {test_loss}]")

    return test_acc
