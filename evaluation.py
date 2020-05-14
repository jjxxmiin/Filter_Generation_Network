import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import FVGG, FResNet18
from lib.models.module import GFLayer
from lib.helper import ClassifyTrainer
from lib.interpretable import GradCAM

plt.interactive(False)


def print_inference_time(model, args):
    total = 0

    for m in model.modules():
        if isinstance(m, GFLayer):
            total += m.weights.data.numel()
        elif isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    criterion = nn.CrossEntropyLoss().to(args.device)

    total_time = 0

    for i in range(0, 5):
        trainer = ClassifyTrainer(model,
                                  criterion,
                                  test_loader=test_loader)

        start = time.time()
        test_loss, test_acc = trainer.test()
        end = time.time()

        test_acc = test_acc / args.batch_size

        total_time += end - start

    print(f'Total Params : {total} \n'
          f' + NUM Filter : {args.num_filters} \n'
          f' + EDGE Filter : {args.edge_filter_type} \n'
          f' + TEXTURE Filter : {args.texture_filter_type} \n'
          f' + OBJECT Filter : {args.object_filter_type} \n'
          f' + Total Time : {total_time / 5} \n'
          f' + TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ] \n')


def print_model_param_nums(model):
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))


def print_model_param_flops(model,
                            input_res=[224, 224],
                            multiply_adds=True,
                            device='cuda'):

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)

    input = torch.autograd.Variable(torch.rand(3, input_res[1], input_res[0]).unsqueeze(0),
                     requires_grad=True)
    out = model(input.to(device))

    total_flops = (sum(list_conv) +
                   sum(list_linear) +
                   sum(list_bn) +
                   sum(list_relu) +
                   sum(list_pooling) +
                   sum(list_upsample))

    print('  + Number of FLOPs: %.3fG' % (total_flops / 1e9))

    return total_flops


def show_grad_cam(model, label, test_loader):
    grad_cam = GradCAM(model, label)

    grad_cam.save_img(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10')

    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_filters', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--edge_filter_type', type=str, default='conv')
    parser.add_argument('--texture_filter_type', type=str, default='normal')
    parser.add_argument('--object_filter_type', type=str, default='normal')
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--log_path', type=str, default='./cifar10.log')

    args = parser.parse_args()

    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    name = f'{args.datasets}_' \
           f'{args.model_name}_' \
           f'{args.num_filters}_' \
           f'{args.edge_filter_type}_' \
           f'{args.texture_filter_type}_' \
           f'{args.object_filter_type}'

    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    filter_types = [args.edge_filter_type,
                    args.texture_filter_type,
                    args.object_filter_type]

    if args.model_name == 'vgg16':
        model = FVGG('VGG16',
                     num_filters=args.num_filters,
                     filter_types=filter_types).to(args.device)

    elif args.model_name == 'resnet18':
        model = FResNet18(filter_types=filter_types).to(args.device)


    model.load_state_dict(torch.load(f'./checkpoint/{name}_model.pth'))

    print_model_param_nums(model)
    print_model_param_flops(model, input_res=[32, 32])
    print_inference_time(model, args)
    # show_grad_cam(model, label, test_loader)