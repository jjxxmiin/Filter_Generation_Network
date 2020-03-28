import os
import re
import torchvision.transforms as transforms
from itertools import combinations

from src.models.vgg import load_model, get_layer_index
from src.prune import *
from src.loader import get_tiny_imagenet_loader
from src.utils import save_pkl, get_logger
from src.benchmark import get_flops
from src.search import Search


def search_prune(model, idx, data_path, subset, check_cls, transformer):
    search = Search(model, data_path, subset, check_cls, transformer=transformer)
    filters = search.get_filter_idx()

    for i, f in enumerate(filters[:-1]):
        idx[i] = idx[i][f]

    model = prune(model, filters)
    flops, params = get_flops(model, input_shape=(3, 64, 64))
    logging.info(f"FLOPs : {flops} / Params : {params}")

    return model, idx

# HyperParam
train_transformer = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])

data_path = './datasets/tiny_imagenet'
batch_size = 32
lr = 0.001

logger = get_logger('./tiny_imagenet_log.log')
class_name = os.listdir('./models/imagenet')

for i in range(len(class_name)):
    part = class_name[i].split('\'')
    class_name[i] = tuple([part[1], part[3], part[5]])

for subset in class_name:
    logger.info(subset)

    model_path = f'./models/imagenet/{subset}.pth'

    for check_idx, check_cls in enumerate(subset):
        logger.info(check_cls)

        idx = get_layer_index()
        model = load_model(model_path, mode='eval')

        train_loader, test_loader = get_tiny_imagenet_loader(data_path,
                                                       subset=subset,
                                                       batch_size=batch_size,
                                                       train_transformer=train_transformer,
                                                       test_transformer=test_transformer)

        model, idx = search_prune(model, idx, data_path, subset, check_cls, transformer=test_transformer)

        for _ in range(0, 10):
            model, train_acc = train(model, train_loader, batch_size, lr)
            test(model, test_loader, batch_size)

        logger.info("Convert Multi -> Binary")
        model = to_binary(model, check_idx)

        for _ in range(0, 5):
            model, idx = search_prune(model, idx, data_path, subset, check_cls, transformer=test_transformer)

            binary_train_loader, binary_test_loader = get_tiny_imagenet_loader(data_path,
                                                                         subset=subset,
                                                                         batch_size=batch_size,
                                                                         train_transformer=train_transformer,
                                                                         test_transformer=test_transformer,
                                                                         true_name=check_cls)

            for _ in range(0, 5):
                model = binary_sigmoid_train(model, binary_train_loader, lr)
                binary_sigmoid_test(model, binary_test_loader)

        save_pkl(idx, f'./pkl/imagenet/{subset}_{check_cls}_idx.pkl')
