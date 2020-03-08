import os
import logging
import torch
import torch.nn as nn
import numpy as np
from src.utils import get_class_path
from PIL import Image
from torch.nn import functional as F


class Search(object):
    def __init__(self,
                 model,
                 data_path,
                 check_cls,
                 transformer,
                 data_size=None):
        """
        :param model       : searching model
        :param data_path   : train / test data root path
        :param check_cls   : Remaining class name
        :param transformer : pytorch transformer
        :param data_size   : using dataset size [if data_size = None is all]
        """

        self.model = model
        self.transformer = transformer

        self.cls_names = os.listdir(data_path)

        # remaining class path
        self.cls_img_paths = None

        # remaining class label
        self.true_labels = None
        self.false_labels = []

        # check class dataset image path
        for i, name in enumerate(self.cls_names):
            if name in check_cls:
                paths = get_class_path(data_path, name)
                np.random.shuffle(paths)

                self.cls_img_paths = paths if data_size is None else paths[:data_size]
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

    def backprop(self, image_path, inverse=None, device='cuda'):
        self.model.zero_grad()

        img = self.transformer(Image.open(image_path))
        img = img.unsqueeze(dim=0).to(device)

        # forward
        output = self.model(img).to(device)
        # acc
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        if pred is not self.cls:
            return None

        if inverse is not None:
            pred = inverse

        print(f"pred: {pred} label {self.cls}")

        one_hot_output = torch.zeros(1, h_x.size()[0]).to('cuda')
        one_hot_output[0][pred] = 1

        output.backward(gradient=one_hot_output)

        grads = self.get_conv_grad()

        return grads

    @staticmethod
    def dif(t, f):
        t = abs(t)
        f = abs(f)

        sum_t = t.reshape(t.shape[0], -1).sum(1)
        sum_f = f.reshape(f.shape[0], -1).sum(1)

        return (sum_f - sum_t) / (sum_t + 1e-5)

    def get_diffs(self):
        total_diff = 0

        for img_path in self.cls_img_paths:
            f_grad = []
            for img in img_path:
                diffs = []

                t_grad = self.backprop(img, cls)

                if t_grad is None:
                    continue

                for i in range(len(diff_labels) - 1):
                    f_grad1 = self.backprop(img, inverse=diff_labels[i])
                    f_grad2 = self.backprop(img, inverse=diff_labels[i + 1])

                    if i == 0:
                        for g1, g2 in zip(f_grad1, f_grad2):
                            f_grad.append(np.maximum(g1, g2))
                    else:
                        print("Not implement")

                for t, f in zip(t_grad, f_grad):
                    diffs.append(self.diff(t, f))

                total_diff += np.array(diffs)

        return total_diff

    def get_binary_diffs(self):
        total_diff = 0

        for image_paths in zip(self.cls_paths[cls]):
            for img in image_paths:
                diffs = []

                t_grad = self.backprop(img, 1)
                f_grad = self.backprop(img, inverse=0)

                if t_grad is None:
                    continue

                for t, f in zip(t_grad, f_grad):
                    diffs.append(self.diff(t, f))

                total_diff += np.array(diffs)

        return total_diff

    def get_filter_idx(self, binary=False):
        logging.info(f"Filter Selection for {self.cls_names[self.true_labels]}")

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

        return filter_idx
