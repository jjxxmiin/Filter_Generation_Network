import argparse
import matplotlib.pyplot as plt
from lib.utils import load_pkl

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--edge_filter_type', type=str, default='sobel')
parser.add_argument('--texture_filter_type', type=str, default='uniform')
parser.add_argument('--object_filter_type', type=str, default='normal')
parser.add_argument('--model_name', type=str, default='resnet18')
parser.set_defaults(feature=True)
args = parser.parse_args()

num_ = 2

fig, axes = plt.subplots(num_, num_, figsize=(10, 10))

dataset = 'cifar10'
t = 'normal'

type = [['conv', 'conv', 'conv'],
        ['sobel', 'normal', 'normal'],
        ['conv', 'normal', 'normal'],
        ['normal', 'normal', 'normal']]

name = f'{dataset}_' \
       f'{args.model_name}_' \
       f'{args.num_filters}_' \
       f'{args.edge_filter_type}_' \
       f'{args.texture_filter_type}_' \
       f'{args.object_filter_type}'

for f in type:
    train_acc = load_pkl(f'./pkl/{dataset}_'
                         f'{args.model_name}'
                         f'{args.num_filters}_'
                         f'{f[0]}_'
                         f'{f[1]}_'
                         f'{f[2]}_'
                         f'train_acc_log')

    train_loss = load_pkl(f'./pkl/{dataset}_'
                          f'{args.model_name}'
                          f'{args.num_filters}_'
                          f'{f[0]}_'
                          f'{f[1]}_'
                          f'{f[2]}_'
                          f'train_loss_log')

    test_acc = load_pkl(f'./pkl/{dataset}_'
                        f'{args.model_name}'
                        f'{args.num_filters}_'
                        f'{f[0]}_'
                        f'{f[1]}_'
                        f'{f[2]}_'
                        f'test_acc_log')

    test_loss = load_pkl(f'./pkl/{dataset}_'
                         f'{args.model_name}'
                         f'{args.num_filters}_'
                         f'{f[0]}_'
                         f'{f[1]}_'
                         f'{f[2]}_'
                         f'test_loss_log')

    axes[0, 0].plot(train_acc, label=f'{f[0]}_{f[1]}_{f[2]}')
    axes[0, 0].set_title("Train Acc")

    axes[1, 0].plot(train_loss, label=f'{f[0]}_{f[1]}_{f[2]}')
    axes[1, 0].set_title("Train Loss")

    axes[0, 1].plot(test_acc, label=f'{f[0]}_{f[1]}_{f[2]}')
    axes[0, 1].set_title("Test Acc")

    axes[1, 1].plot(test_loss, label=f'{f[0]}_{f[1]}_{f[2]}')
    axes[1, 1].set_title("Test Loss")

for i in range(num_):
    for j in range(num_):
        axes[i, j].legend()
        axes[i, j].grid()

plt.savefig('./origin.png')
plt.show()