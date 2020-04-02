import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from src.utils import get_class_path


class Loader(object):
    def __init__(self,
                 data_path):

        self.data_path = data_path

    def get_loader(self, transformer, batch_size, dtype='train', shuffle=True):
        dataset = datasets.ImageFolder(os.path.join(self.data_path, dtype), transform=transformer)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

        return loader


"""
########################################################################################################################
########################################################################################################################
######################################################## CIFAR10 #######################################################
########################################################################################################################
########################################################################################################################

image shape : 32 x 32
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""


# def sub_cifar(data_path, subset, batch_size, transformer, dtype='train'):
#     sub_dataset = datasets.ImageFolder(os.path.join(data_path, dtype), transform=transformer)
#
#     for s in subset:
#         idx = torch.tensor(sub_dataset.targets) == sub_dataset.classes.index(s)
#
#     sub_dataset = torch.utils.data.Subset(sub_dataset, np.where(idx == 1)[0])
#     sub_loader = data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=True)
#
#     return sub_loader


class Sub_CIFAR(object):
    """
    cifar10 : subset multi
    """

    def __init__(self,
                 data_path,
                 subset,
                 dtype,
                 transformer=None):

        """
        :param data_path   : Dataset Root Path
        :param subset      : [(str) class_1 name, (str) class_2 name ...]
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(subset):
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
    cifar10 : subset binary
    """

    def __init__(self,
                 data_path,
                 subset,
                 true_name,
                 dtype='train',
                 transformer=None):
        """
        :param data_path   : Dataset Root Path
        :param subset      : [(str) class_1_name, (str) class_2_name ...]
        :param true_name   : true class name
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(subset):
            class_path = get_class_path(data_path, dtype, name)

            if name in true_name:
                self.img_path += class_path
                self.labels += [1] * len(class_path)
            else:
                split = int(len(class_path) / (len(subset) - 1))

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


class One_CIFAR(object):
    """
    cifar10 : only one class
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

"""
########################################################################################################################
########################################################################################################################
##################################################### Tiny-ImageNet ####################################################
########################################################################################################################
########################################################################################################################
"""


class Sub_Tiny_ImageNet(object):
    """
    ImageNet : subset multi
    """

    def __init__(self,
                 data_path,
                 subset,
                 dtype='train',
                 transformer=None):

        """
        :param data_path   : Dataset Root Path
        :param subset      : [(str) class_1 name, (str) class_2 name ...]
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        if dtype == 'train':
            for i, name in enumerate(subset):
                class_path = get_class_path(data_path, dtype, os.path.join(name, 'images'))
                self.img_path += class_path
                self.labels += [i] * len(class_path)
        else:
            for i, name in enumerate(subset):
                class_path = get_class_path(data_path, dtype, name)
                self.img_path += class_path
                self.labels += [i] * len(class_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])

        if img.getbands()[0] == 'L':
            img = img.convert('RGB')

        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)


class Sub_Binary_Tiny_ImageNet(object):
    """
    ImageNet : subset binary
    """

    def __init__(self,
                 data_path,
                 subset,
                 true_name,
                 dtype='train',
                 transformer=None):
        """
        :param data_path   : Dataset Root Path
        :param subset      : [(str) class_1_name, (str) class_2_name ...]
        :param true_name   : true class name
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(subset):
            if dtype == 'train':
                class_path = get_class_path(data_path, dtype, os.path.join(name, 'images'))
            else:
                class_path = get_class_path(data_path, dtype, name)

            if name in true_name:
                self.img_path += class_path
                self.labels += [1] * len(class_path)
            else:
                split = int(len(class_path) / (len(subset) - 1))

                self.img_path += class_path[:split]
                self.labels += [0] * split

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if img.getbands()[0] == 'L':
            img = img.convert('RGB')

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)


class One_Tiny_ImageNet(object):
    """
    ImageNet : only one class
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
        """

        self.labels = []
        self.transformer = transformer
        if dtype == 'train':
            self.img_path = get_class_path(data_path, dtype, os.path.join(check_cls, 'images'))
        else:
            self.img_path = get_class_path(data_path, dtype, check_cls)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])

        if img.getbands()[0] == 'L':
            img = img.convert('RGB')

        if self.transformer is not None:
            img = self.transformer(img)

        return img

    def __len__(self):
        return len(self.img_path)


"""
########################################################################################################################
########################################################################################################################
###################################################### GET Loader ######################################################
########################################################################################################################
########################################################################################################################
"""


def get_cifar10_loader(data_path, subset, batch_size, train_transformer, test_transformer, true_name=None):
    if true_name is None:
        train_datasets = Sub_CIFAR(data_path=data_path,
                                   subset=subset,
                                   dtype='train',
                                   transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_CIFAR(data_path=data_path,
                                  subset=subset,
                                  dtype='test',
                                  transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    else:
        train_datasets = Sub_Binary_CIFAR(data_path=data_path,
                                          subset=subset,
                                          dtype='train',
                                          true_name=true_name,
                                          transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_Binary_CIFAR(data_path=data_path,
                                         subset=subset,
                                         dtype='test',
                                         true_name=true_name,
                                         transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    return train_loader, test_loader


def get_tiny_imagenet_loader(data_path, subset, batch_size, train_transformer, test_transformer, true_name=None):
    if true_name is None:
        train_datasets = Sub_Tiny_ImageNet(data_path=data_path,
                                           subset=subset,
                                           dtype='train',
                                           transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_Tiny_ImageNet(data_path=data_path,
                                          subset=subset,
                                          dtype='val',
                                          transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    else:
        train_datasets = Sub_Binary_Tiny_ImageNet(data_path=data_path,
                                                  subset=subset,
                                                  dtype='train',
                                                  true_name=true_name,
                                                  transformer=train_transformer)

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_datasets = Sub_Binary_Tiny_ImageNet(data_path=data_path,
                                                 subset=subset,
                                                 dtype='val',
                                                 true_name=true_name,
                                                 transformer=test_transformer)

        test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    return train_loader, test_loader
