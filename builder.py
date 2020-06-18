import torch
import time
from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import fvgg16_bn
from lib.helper import ClassifyTrainer
from lib.models.module import get_filter, GFLayer

device = 'cuda'
model_path = "./checkpoint/cifar10_vgg16_3_conv_normal_normal_model.pth"
num_filter = 3
batch_size = 64

torch.manual_seed(20145170)
torch.cuda.manual_seed(20145170)

first_filters = get_filter('conv', num_filters=num_filter)
middle_filters = get_filter('normal', num_filters=num_filter)
last_filters = get_filter('normal', num_filters=num_filter)

filters = [first_filters, middle_filters, last_filters]

model = fvgg16_bn(filters=filters).to(device)

model.load_state_dict(torch.load(model_path))

current_layer = 0

start_time = time.time()
for i, (name, module)in enumerate(model.features.named_modules()):
    if isinstance(module, GFLayer):
        current_layer += 1

        in_channels = module.in_ch
        out_channels = module.out_ch
        groups = module.groups
        stride = module.stride
        padding = module.padding

        if current_layer <= 8:
            f = middle_filters
        else:
            f = last_filters

        new_weights = f.view(1, 1, 3, 3, 3) * \
            module.weights.view(out_channels, in_channels // groups, 3, 1, 1).repeat(1, 1, 1, 3, 3)

        new_weights = new_weights.sum(2)

        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=padding,
                                   groups=groups,
                                   bias=(module.bias is not None)).to(device)

        new_conv.weight.data = new_weights
        model.features[i-1] = new_conv
endtime = time.time()

print(f"Require Build Time : {endtime - start_time}")

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_dataset = datasets.CIFAR10(root='../data',
                                train=False,
                                transform=test_transformer,
                                download=True)

test_loader = utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

criterion = nn.CrossEntropyLoss().to(device)

test_iter = len(test_loader)

trainer = ClassifyTrainer(model,
                          criterion,
                          train_loader=None,
                          test_loader=test_loader,
                          optimizer=None,
                          scheduler=None)

best_test_acc = 0

test_loss, test_top1_acc, _ = trainer.test()
test_acc = test_top1_acc / batch_size

print(f"Top1 Acc : {test_acc} Loss : {test_loss}")