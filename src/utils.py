import os
import shutil
import logging
import pickle
from logging import handlers

def get_logger(file_name='cifar10_log.log'):
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


def get_class_path(data_path, dtype, class_name):
    """
    :param data_path    : dataset root path
    :param dtype        : train or test
    :param class_name   : class name

    :return: {root_path}/{dtype}/{class_name} paths
    """
    result = []

    type_path = os.path.join(data_path, dtype)
    class_path = os.path.join(type_path, class_name)

    for img_name in os.listdir(class_path):
        result.append(os.path.join(class_path, img_name))

    return result


def save_pkl(data, path):
    logging.info("SUCESS Save")

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    logging.info("SUCESS Load")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def name_to_label(data_path, dtype='train', label_file='words.txt'):
    class_table = {}

    with open(os.path.join(data_path, label_file), 'r') as f:
        word_to_label = f.readlines()

        for wtl in word_to_label:
            label, word = wtl.rstrip('\n').split('\t')
            name = word.split(',')[0]

            class_table[label] = name

    path = os.path.join(data_path, dtype)
    label = os.listdir(path)

    for i in label:
        if not os.path.isdir(os.path.join(path, i)):
            pass

        os.rename(os.path.join(path, i), os.path.join(path, class_table[i]))


def fit_structure(data_path):
    name_to_label(data_path, dtype='train')

    val_data_path = os.path.join(data_path, 'val')

    with open(os.path.join(val_data_path, 'val_annotations.txt')) as f:
        word_to_label = f.readlines()

        for wtl in word_to_label:
            img_name, label = wtl.rstrip('\n').split('\t')[:2]

            if not os.path.exists(os.path.join(val_data_path, label)):
                os.mkdir(os.path.join(val_data_path, label))

            shutil.move(os.path.join(val_data_path, f'images/{img_name}'),
                        os.path.join(val_data_path, f'{label}/{img_name}'), )

    os.removedirs(os.path.join(val_data_path, 'images'))

    name_to_label(data_path, dtype='val')

