import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.vgg import load_model
from src.loader import Sub_CIFAR

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

subset = ('airplane', 'automobile', 'truck')


test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_datasets = Sub_CIFAR(configs['data_path'],
                          subset,
                          dtype='test',
                          transformer=test_transformer)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=configs['batch_size'],
                                          shuffle=True)

# model
model_path = f'./models/cifar10/VGG16_{subset}.pth'
model = load_model(model_path, mode='eval')
print(model)

# cost
criterion = nn.CrossEntropyLoss().to(device)

test_iter = len(test_loader)
test_loss = 0
n_test_correct = 0

for images, labels in test_loader:
    images, label = images.to(device), labels.to(device)

    pred = model(images)
    # acc
    _, predicted = torch.max(pred, 1)


    n_test_correct += (predicted == labels).sum().item()
    # loss
    loss = criterion(pred, labels)
    test_loss += loss.item()

test_acc = n_test_correct / (test_iter * configs['batch_size'])
test_loss = test_loss / test_iter

print(f" TEST [Acc / Loss] : [ {test_acc} / {test_loss} ]")
