import torch
import torch.nn as nn
import numpy as np
import random
from src.loader import One_CIFAR
import logging


class Search(object):
    def __init__(self,
                 model,
                 data_path,
                 subset,
                 check_cls,
                 transformer,
                 dtype='train',
                 prog=None):
        """
        :param model       : searching model
        :param data_path   : train / test data root path
        :param subset      : [(str)class_1 name, (str)class_2 name ...]
        :param check_cls   : Remaining class name
        :param transformer : pytorch transformer
        :param dtype       : train, test
        """

        self.model = model

        # remaining class label
        self.true_labels = None
        self.false_labels = []
        self.prog = prog

        datasets = One_CIFAR(data_path=data_path,
                           dtype=dtype,
                           check_cls=check_cls,
                           transformer=transformer)

        self.loader = torch.utils.data.DataLoader(dataset=datasets,
                                                  batch_size=32,
                                                  shuffle=True)

        # check class dataset image path
        for i, name in enumerate(subset):
            if name == check_cls:
                self.true_labels = i
            else:
                self.false_labels.append(i)

        logging.info(f"true labels : {self.true_labels}")
        logging.info(f"false labels : {self.false_labels}")

    def get_conv_grad(self):
        grads = []

        for m in self.model.modules():
            if type(m) == nn.Conv2d:
                grads.append(m.weight.grad.cpu().detach().numpy())

        return grads

    def get_feature_hook(self, module, input, output):
        self.features.append(output)

    def get_gradient_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0])

    def register(self):
        for module, (name, _) in zip(self.model.features.modules(), self.model.features.named_modules()):
            if type(module) == nn.Conv2d:
                module.register_forward_hook(self.get_feature_hook)
                module.register_backward_hook(self.get_gradient_hook)

    def backprop(self, img, cls, device='cuda'):
        self.model.zero_grad()

        img = img.to(device)
        # forward
        output = self.model(img).to(device)
        # acc
        _, pred = torch.max(output, 1)

        if output.size()[1] == 1:
            one_hot_output = torch.full(output.size(), cls).to(device)
        else:
            one_hot_output = torch.zeros(output.size()).to(device)
            one_hot_output[:, cls] = 1

        output.backward(gradient=one_hot_output)

        grads = self.get_conv_grad()

        return grads

    def get_diffs(self):
        total_diff = 0

        # true image
        for idx, img in enumerate(self.loader):
            if self.prog is not None:
                self.prog.setValue(100 / len(self.loader) * (idx + 1))

            diffs = []

            t_grad = self.backprop(img, cls=self.true_labels)
            f_grad = self.backprop(img, cls=self.false_labels[0])

            sum_t_grad = [abs(t.reshape(t.shape[0], -1)).sum(1) for t in t_grad]
            sum_f_grad = [abs(f.reshape(f.shape[0], -1)).sum(1) for f in f_grad]

            for f_label in self.false_labels[1:]:
                f_grad_next = self.backprop(img, cls=f_label)
                sum_f_next_grad = [abs(f.reshape(f.shape[0], -1)).sum(1) for f in f_grad_next]

                for i, (sum_f1, sum_f2) in enumerate(zip(sum_f_grad, sum_f_next_grad)):
                    sum_f_grad[i] = np.maximum(sum_f1, sum_f2)

            for sum_t, sum_f in zip(sum_t_grad, sum_f_grad):
                diffs.append(sum_t - sum_f)

            total_diff += np.array(diffs)

        return total_diff

    def get_binary_diffs(self):
        total_diff = 0

        for idx, img in enumerate(self.loader):
            diffs = []

            t_grad = self.backprop(img, cls=1)
            f_grad = self.backprop(img, cls=0)

            sum_t_grad = [abs(t.reshape(t.shape[0], -1)).sum(1) for t in t_grad]
            sum_f_grad = [abs(f.reshape(f.shape[0], -1)).sum(1) for f in f_grad]

            for sum_t, sum_f in zip(sum_t_grad, sum_f_grad):
                diffs.append(sum_t - sum_f)

            total_diff += np.array(diffs)

        return total_diff

    def get_filter_idx(self, binary=False):
        if binary:
            diffs = self.get_binary_diffs()
        else:
            diffs = self.get_diffs()

        filter_idx = [[] for _ in range(len(diffs))]

        for i, diff in enumerate(diffs):
            for j, d in enumerate(diff):
                if i < 2:
                    filter_idx[i].append(j)
                else:
                    if d > 0:
                        filter_idx[i].append(j)

            if not filter_idx[i]:
                idx = [k for k in range(len(diff))]

                filter_idx[i] = random.sample(idx, int(np.ceil(len(idx) * 0.9)))

        return filter_idx


def get_random_filter_idx(filter_idx):
    for i, idx in enumerate(filter_idx):
        filter_idx[i] = random.sample(idx, int(np.ceil(len(idx) * 0.9)))

    return filter_idx
