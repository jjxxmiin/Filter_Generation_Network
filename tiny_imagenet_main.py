import os
import torchvision.transforms as transforms
from itertools import combinations

from src.benchmark import get_flops
from src.models.vgg import load_model, get_layer_index
from src.prune import *
from src.loader import get_cifar10_loader
from src.search import Search
from src.utils import save_pkl, name_to_label

import logging
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


def search_prune(model, idx, data_path, subset, check_cls, transformer):
    search = Search(model, data_path, subset, check_cls, transformer=transformer)
    filters = search.get_filter_idx()

    for i, f in enumerate(filters[:-1]):
        idx[i] = idx[i][f]

    model = prune(model, filters)
    flops, params = get_flops(model)
    logging.info(f"FLOPs : {flops} / Params : {params}")

    return model, idx


data_path = './datasets/tiny_imagenet'
# name_to_label(data_path)

Load
