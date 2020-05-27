import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from torch import utils
from torchvision import datasets, transforms
from lib.models.cifar10 import fvgg16_bn, fresnet18
from lib.models.cifar100 import mobilenetv2, shufflenetv2
from lib.utils import load_pkl, print_inference_time, print_model_param_flops, print_model_param_nums

plt.interactive(False)

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--datasets', type=str, default='cifar10')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test', action='store_true')
parser.add_argument('--edge_filter_type', type=str, default='conv')
parser.add_argument('--texture_filter_type', type=str, default='normal')
parser.add_argument('--object_filter_type', type=str, default='normal')
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--log_path', type=str, default='./cifar10.log')

args = parser.parse_args()

colourWheel =['#329932',
              '#ff6961',
              'b',
              '#6a3d9a',
              '#fb9a99',
              '#e31a1c',
              '#fdbf6f',
              '#ff7f00',
              '#cab2d6',
              '#6a3d9a',
              '#ffff99',
              '#b15928',
              '#67001f',
              '#b2182b',
              '#d6604d',
              '#f4a582',
              '#fddbc7',
              '#f7f7f7',
              '#d1e5f0',
              '#92c5de',
              '#4393c3',
              '#2166ac',
              '#053061']

type = [['conv', 'conv', 'conv'],
        ['conv', 'uniform', 'uniform'],
        ['conv', 'exp', 'exp'],
        ['conv', 'normal', 'uniform'],
        ['conv', 'uniform', 'normal'],
        ['conv', 'normal', 'normal'],
        ['normal', 'normal', 'normal'],
        ['sobel', 'normal', 'normal'],
        ['line', 'normal', 'normal']]

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, t in enumerate(type):
    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    name = f'{args.datasets}_' \
           f'{args.model_name}_' \
           f'{args.num_filters}_' \
           f'{t[0]}_' \
           f'{t[1]}_' \
           f'{t[2]}_'

    train_acc = load_pkl(f'./pkl/{name}'
                         f'_train_acc_log')

    train_loss = load_pkl(f'./pkl/{name}'
                          f'_train_loss_log')

    test_acc = load_pkl(f'./pkl/{name}'
                        f'_test_acc_log')

    test_loss = load_pkl(f'./pkl/{name}'
                         f'_test_loss_log')

    axes[0].plot(np.array(train_acc) * 100,
                 color=colourWheel[i],
                 label=f'{"_".join(t)}')
    axes[0].set_title("Train Acc")

    axes[1].plot(train_loss,
                 color=colourWheel[i],
                 label=f'{"_".join(t)}')
    axes[1].set_title("Train Loss")

    axes[2].plot(np.array(test_acc) * 100,
                 color=colourWheel[i],
                 label=f'{"_".join(t)}')
    axes[2].set_title("Test Acc")

    axes[3].plot(test_loss,
                 color=colourWheel[i],
                 label=f'{"_".join(t)}')
    axes[3].set_title("Test Loss")

plt.show()

acc_log = []
param_log = []
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, t in enumerate([type[3]]):
    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    name = f'{args.datasets}_' \
           f'{args.model_name}_' \
           f'{args.num_filters}_' \
           f'{t[0]}_' \
           f'{t[1]}_' \
           f'{t[2]}_'

    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    if args.model_name == 'vgg16':
        model = fvgg16_bn(filter_types=t, num_filters=args.num_filters).to(args.device)

    elif args.model_name == 'resnet18':
        model = fresnet18(filter_types=t).to(args.device)

    model.load_state_dict(torch.load(f'./checkpoint/{name}_model.pth'))

    param = print_model_param_nums(model)
    print_model_param_flops(model, input_res=[32, 32])
    acc = print_inference_time(model, args, test_loader)
    # show_grad_cam(model, label, test_loader)

    acc_log.append(acc)
    param_log.append(param)

axes[0].plot(np.array(acc_log) * 100,
             color=colourWheel[13])
axes[0].set_title("Accuracy")
axes[0].grid()
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xlabel('Num Filters')

axes[1].plot(param_log,
             color=colourWheel[14])
axes[1].set_title("Params")
axes[1].grid()
axes[1].set_ylabel('Params (M)')
axes[1].set_xlabel('Num Filters')

plt.show()