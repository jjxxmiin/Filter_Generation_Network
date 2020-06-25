import os
import time
import torch
import argparse
from torch import nn, utils
from torchvision import datasets, transforms
from lib.helper import ClassifyTrainer
from lib.models.module import get_filter, GFLayer
from lib.models.cifar10 import fvgg16_bn
from lib.utils import print_model_param_nums

parser = argparse.ArgumentParser(description='Builder')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--datasets', type=str, default='cifar10')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--edge_filter_type', '-e', type=str, default='conv')
parser.add_argument('--texture_filter_type', '-t', type=str, default='normal')
parser.add_argument('--object_filter_type', '-o', type=str, default='normal')
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--seed', type=int, default=20145170)
parser.add_argument('--test', action='store_true')
parser.set_defaults(feature=True)
args = parser.parse_args()

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# model path
name = f'{args.datasets}_' \
       f'{args.model_name}_' \
       f'{args.num_filters}_' \
       f'{args.edge_filter_type}_' \
       f'{args.texture_filter_type}_' \
       f'{args.object_filter_type}_model.pth'

model_path = os.path.join(args.save_path, name)

# get filter
first_filters = get_filter(args.edge_filter_type, num_filters=args.num_filters)
middle_filters = get_filter(args.texture_filter_type, num_filters=args.num_filters)
last_filters = get_filter(args.object_filter_type, num_filters=args.num_filters)

filters = [first_filters, middle_filters, last_filters]

# load model
model = fvgg16_bn(filters=filters).to(args.device)

model.load_state_dict(torch.load(model_path))

# print param
print_model_param_nums(model)

current_layer = 0
start_time = time.time()

# build
for i, (name, module) in enumerate(model.features.named_modules()):
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
                                   bias=(module.bias is not None)).to(args.device)

        new_conv.weight.data = new_weights
        model.features[i-1] = new_conv

endtime = time.time()

print(f"  + Require Build Time : {endtime - start_time}")

# print param
print_model_param_nums(model)

if args.test:
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    criterion = nn.CrossEntropyLoss().to(args.device)

    test_iter = len(test_loader)

    trainer = ClassifyTrainer(model,
                              criterion,
                              train_loader=None,
                              test_loader=test_loader,
                              optimizer=None,
                              scheduler=None)

    best_test_acc = 0

    test_loss, test_top1_acc, _ = trainer.test()
    test_acc = test_top1_acc / args.batch_size

    print(f"Top1 Acc : {test_acc} Loss : {test_loss}")