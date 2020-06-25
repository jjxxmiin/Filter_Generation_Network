import torch
import torch.nn as nn
import numpy as np
import pickle
import time
import logging
from logging import handlers
from lib.models.module import GFLayer
from lib.helper import ClassifyTrainer
from lib.interpretable import GradCAM


def get_logger(file_name='log.log'):
    # create logger
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # formatter handler
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    # file handler
    log_max_size = 10 * 1024 * 1024
    log_file_count = 20

    file_handler = handlers.RotatingFileHandler(filename=file_name, maxBytes=log_max_size, backupCount=log_file_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def print_inference_time(model, args, test_loader):
    total = 0

    for m in model.modules():
        if isinstance(m, GFLayer):
            total += m.weights.data.numel()
        elif isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    criterion = nn.CrossEntropyLoss().to(args.device)

    total_time = 0

    trainer = ClassifyTrainer(model,
                              criterion,
                              test_loader=test_loader)

    start = time.time()
    test_loss, test_top1_acc, test_top5_acc = trainer.test()
    end = time.time()

    test_top1_acc = test_top1_acc / args.batch_size

    total_time += end - start

    print(f'Total Params : {total} \n'
          f' + NUM Filter : {args.num_filters} \n'
          f' + EDGE Filter : {args.edge_filter_type} \n'
          f' + TEXTURE Filter : {args.texture_filter_type} \n'
          f' + OBJECT Filter : {args.object_filter_type} \n'
          f' + Total Time : {total_time / args.batch_size} \n'
          f' + TEST  [Loss / Top1@Acc / Top5@Acc] : [ {test_loss} / {test_top1_acc} / {test_top5_acc}] \n')

    return test_top1_acc, test_top5_acc


def print_model_param_nums(model):
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))

    return total


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

    list_gf = []

    def gf_hook(self, input, output):
        kernel_size = 3

        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = kernel_size * kernel_size * (self.in_ch / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # conv flops
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        # gen flops
        flops += kernel_size * kernel_size * 3 * output_channels * input_channels * batch_size

        list_gf.append(flops)

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
            if isinstance(net, GFLayer):
                net.register_forward_hook(gf_hook)
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
                   sum(list_upsample) +
                   sum(list_gf))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 1e9))

    return total_flops


def show_grad_cam(model, label, test_loader):
    grad_cam = GradCAM(model, label)

    grad_cam.save_img(test_loader)