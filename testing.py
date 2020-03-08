import torch
import matplotlib.pyplot as plt
from src.utils import *
from src.models.vgg import VGG, get_layer_index
from src.search import Search
from src.prune import *


configs = {
    'task': 'classify',
    'model': 'VGG16',
    'dataset': 'CIFAR10',
    'classes': 10,
    'batch_size': 32,
    'lr': 0.001,
    'root_path': './datasets/cifar10'
}

logger = get_logger('./log.log')
class_names = ['airplane', 'dog', 'horse']

# model
model = VGG('VGG16').to('cuda')
model.load_state_dict(torch.load(f"./{configs['model']}_{configs['dataset']}.pth"))
model.eval()

idx = get_layer_index('VGG16')
# class 0 : airplane
c = 0

for _ in range(0, 1):
    # filter search
    search = Search(model,
                    configs['root_path'],
                    class_names,
                    data_size=5000,
                    dtype='train')

    filters = search.get_filter_idx(c)

    # pruning
    model = prune(model, filters=filters)

    # train / test
    for _ in range(0, 10):
        model = train(model, configs, logger=logger)
        test(model, configs, logger=logger)

# multi to binary
model = to_binary(model, c)

for _ in range(0, 5):
    # binary search
    search = Search(model,
                    configs['root_path'],
                    class_names,
                    data_size=5000,
                    dtype='train')

    filters = search.get_filter_idx(c, binary=True)

    # pruning
    model = prune(model, filters=filters)

    # binary train / test
    for _ in range(0, 5):
        model = binary_train(model, configs, c=c, logger=logger)
        binary_test(model, configs, c=c, logger=logger)
