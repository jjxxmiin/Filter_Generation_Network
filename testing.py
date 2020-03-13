import torchvision.transforms as transforms

from src.benchmark import get_flops
from src.models.vgg import load_model, get_layer_index
from src.prune import *
from src.loader import get_train_test_loader
from src.search import Search

import logging
import pickle
from logging import handlers


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


# TODO : Dynamic for pytorch transforms
train_transformer = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])

cls = ['dog', 'airplane',  'horse']
model_path = './VGG16_CIFAR10.pth'
data_path = './datasets/cifar10'
batch_size = 32
lr = 0.001
logger = get_logger('log.log')


def search_prune(model, idx, data_path, check_cls, transformer):
    search = Search(model, data_path, check_cls, transformer=transformer)
    filters = search.get_filter_idx()

    for i, f in enumerate(filters[:-1]):
        idx[i] = idx[i][f]

    model = prune(model, filters)
    flops, params = get_flops(model)
    logger.info(f"FLOPs : {flops} / Params : {params}")

    return model, idx


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    return data


for check_idx, check_cls in enumerate(cls):
    idx = get_layer_index()
    model = load_model(model_path, mode='eval')

    train_loader, test_loader = get_train_test_loader(data_path,
                                                      batch_size=batch_size,
                                                      train_transformer=train_transformer,
                                                      test_transformer=test_transformer)

    for _ in range(0, 1):
        model, idx = search_prune(model, idx, data_path, check_cls, transformer=test_transformer)

        for _ in range(0, 10):
            model, train_acc = train(model, train_loader, batch_size, lr)
            test(model, test_loader, batch_size)

    logger.info("Convert Multi -> Binary")
    model = to_binary(model, check_idx)

    train_loader, test_loader = get_train_test_loader(data_path,
                                                      batch_size=batch_size,
                                                      train_transformer=train_transformer,
                                                      test_transformer=test_transformer,
                                                      true_name=check_cls)

    for _ in range(0, 5):
        model, idx = search_prune(model, idx, data_path, check_cls, transformer=test_transformer)

        for _ in range(0, 5):
            model = binary_sigmoid_train(model, train_loader, lr)
            binary_sigmoid_test(model, test_loader)

    save_pkl(idx, f'./{check_cls}_idx.pkl')

# for check_idx, check_cls in enumerate(cls):
#     idx = get_layer_index()
#     model = load_model(model_path, mode='eval')
#
#     model = to_binary(model, check_idx)
#
#     train_loader, test_loader = get_train_test_loader(data_path,
#                                                       batch_size=batch_size,
#                                                       train_transformer=train_transformer,
#                                                       test_transformer=test_transformer,
#                                                       true_name=check_cls)
#
#     for _ in range(0, 20):
#         model, idx = search_prune(model, idx, data_path, check_cls, transformer=test_transformer)
#
#         for _ in range(0, 5):
#             model = binary_sigmoid_train(model, train_loader, lr)
#             binary_sigmoid_test(model, test_loader)
#
#     save_pkl(idx, f'./{check_cls}_binary_idx.pkl')

