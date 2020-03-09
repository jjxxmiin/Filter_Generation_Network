import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from src.loader import Class_CIFAR
from src.models.vgg import load_model


class Search(object):
    def __init__(self,
                 model,
                 data_path,
                 check_cls,
                 transformer,
                 dtype='train'):
        """
        :param model       : searching model
        :param data_path   : train / test data root path
        :param check_cls   : Remaining class name
        :param transformer : pytorch transformer
        :param data_size   : using dataset size [if data_size = None is all]
        """

        self.model = model
        self.cls_names = os.listdir(os.path.join(data_path, dtype))

        # remaining class path
        self.cls_img_paths = None

        # remaining class label
        self.true_labels = None
        self.false_labels = []

        datasets = Class_CIFAR(data_path=data_path,
                               dtype='train',
                               cls_name=check_cls,
                               transformer=transformer)

        self.loader = torch.utils.data.DataLoader(dataset=datasets,
                                             batch_size=32,
                                             shuffle=True)

        # check class dataset image path
        for i, name in enumerate(self.cls_names):
            if name in check_cls:
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

    def backprop(self, img, cls, device='cuda'):
        self.model.zero_grad()
        # forward
        output = self.model(img).to(device)
        # acc
        _, pred = torch.max(output, 1)
        one_hot_output = torch.zeros(output.size()[0], 3).to('cuda')

        one_hot_output[:, cls] = 1
        output.backward(gradient=one_hot_output)

        grads = self.get_conv_grad()

        return grads

    @staticmethod
    def diff(t, f):
        t = abs(t)
        f = abs(f)

        sum_t = t.reshape(t.shape[0], -1).sum(1)
        sum_f = f.reshape(f.shape[0], -1).sum(1)

        return (sum_f - sum_t) / (sum_t + 1e-5)

    def get_diffs(self):
        total_diff = 0

        # true image
        for idx, img in enumerate(self.loader):
            img = img.to('cuda')
            diffs = []
            # all t_grad
            t_grad = self.backprop(img, cls=self.true_labels)
            f_grad = self.backprop(img, cls=self.false_labels[0])

            for i in range(1, len(self.false_labels)):
                f_grad_next = self.backprop(img, cls=self.false_labels[i])

                for j in range(len(f_grad)):
                    f_grad[j] = np.maximum(f_grad[j], f_grad_next[j])

            for t, f in zip(t_grad, f_grad):
                diffs.append(self.diff(t, f))

            total_diff += np.array(diffs)

        return total_diff


model = load_model("./VGG16_CIFAR10.pth")

transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])
data_path = './datasets/cifar10'

# 1 stage
search = Search(model,
                data_path,
                ['airplane'],
                transformer=transformer,
                dtype='train')

a = search.get_diffs()

