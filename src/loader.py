import os
import torch
import torchvision
from torch.utils.data import sampler
from PIL import Image
from src.utils import get_class_path


class CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        if mode == 'train':
            train = True
        else:
            train = False

        cifar10_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                       train=train,
                                                       transform=transformer,
                                                       download=True)

        cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=shuffle)

        return cifar10_loader


class Sub_CIFAR(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 data_path,
                 dtype,
                 transformer=None):

        """
        :param data_path   : Dataset Root Path
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(os.listdir(os.path.join(data_path, dtype))):
            class_path = get_class_path(data_path, dtype, name)
            self.img_path += class_path
            self.labels += [i] * len(class_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)


class Sub_Binary_CIFAR(object):
    """
    image shape : 32 x 32
    """

    def __init__(self,
                 data_path,
                 true_name,
                 dtype='train',
                 transformer=None):
        """
        :param data_path   : Dataset Root Path
        :param class_names : part of cifar10 class name
        :param dtype       : train, test
        :param c           : class 1 / other 0
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        cls_names = os.listdir(os.path.join(data_path, dtype))

        for i, name in enumerate(cls_names):
            class_path = get_class_path(data_path, dtype, name)

            if name in true_name:
                self.img_path += class_path
                self.labels += [1] * len(class_path)
            else:
                split = int(len(class_path) / (len(cls_names) - 1))

                self.img_path += class_path[:split]
                self.labels += [0] * split

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)


class Class_CIFAR(object):
    """
    image shape : 32 x 32
    """

    def __init__(self,
                 data_path,
                 check_cls,
                 dtype='train',
                 transformer=None):
        """
        :param data_path   : Dataset Root Path
        :param check_cls   : cifar10 class name
        :param dtype       : train, test
        :param c           : class 1 / other 0
        """

        self.labels = []
        self.transformer = transformer
        self.img_path = get_class_path(data_path, dtype, check_cls)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])

        if self.transformer is not None:
            img = self.transformer(img)

        return img

    def __len__(self):
        return len(self.img_path)


def get_train_test_loader(data_path, batch_size, train_transformer, test_transformer, true_name=None):
    if true_name is None:
        train_datasets = Sub_CIFAR(data_path=data_path,
                                   dtype='train',
                                   transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_CIFAR(data_path=data_path,
                                  dtype='test',
                                  transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)


    else:
        train_datasets = Sub_Binary_CIFAR(data_path=data_path,
                                          dtype='train',
                                          true_name=true_name,
                                          transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_Binary_CIFAR(data_path=data_path,
                                         dtype='test',
                                         true_name=true_name,
                                         transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    return train_loader, test_loader
