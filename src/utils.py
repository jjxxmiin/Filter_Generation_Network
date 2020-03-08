import os
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


def get_class_path(data_path, class_name):
    """
    :param data_path    : dataset root path
    :param class_name   : class name

    :return: {root_path}/{dtype}/{class_name} paths
    """
    result = []

    class_path = os.path.join(data_path, class_name)

    for img_name in os.listdir(class_path):
        result.append(os.path.join(class_path, img_name))

    return result


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
