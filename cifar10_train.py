import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src.models.vgg import VGG
from src.loader import Loader

sys.path.append('.')

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'classify',
    'model': 'VGG16',
    'dataset': 'CIFAR10',
    'classes': 10,
    'lr': 0.01,
    'epochs': 100,
    'batch_size': 32,
    'data_path': './datasets/cifar10'
}

# augmentation
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(size=(32, 32), padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# datasets/loader
loader = Loader(configs['data_path'])

train_loader = loader.get_loader(transformer=train_transformer,
                                 batch_size=configs['batch_size'],
                                 dtype='train',
                                 shuffle=True,)

test_loader = loader.get_loader(transformer=test_transformer,
                                batch_size=configs['batch_size'],
                                dtype='test',
                                shuffle=True,)

# model
model = VGG('VGG16', output=configs['classes']).to(device)
print(model)

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
        torch.save(model.state_dict(), f"./models/{configs['model']}.pth")
        best_valid_acc = valid_acc
