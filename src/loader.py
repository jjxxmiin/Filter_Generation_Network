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
                 root_path,
                 class_names,
                 dtype='train',
                 transformer=None):

        """
        :param root_path   : Dataset Root Path
        :param class_names : part of cifar10 class name
        :param dtype       : train, test
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, cls in enumerate(class_names):
            class_path = get_class_path(root_path, dtype, cls)
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
                 root_path,
                 class_names,
                 c,
                 dtype='train',
                 transformer=None):
        """
        :param root_path   : Dataset Root Path
        :param class_names : part of cifar10 class name
        :param dtype       : train, test
        :param c           : class 1 / other 0
        """

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(class_names):
            class_path = get_class_path(root_path, dtype, name)
            self.img_path += class_path

            if i == c:
                self.labels += [1] * len(class_path)
            else:
                self.labels += [0] * len(class_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)
