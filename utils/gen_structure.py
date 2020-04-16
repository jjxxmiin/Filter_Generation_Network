import os
import tarfile
import argparse
from tqdm import tqdm


def trainprep(path):
    train_path = os.path.join(path, "train")

    tar_list = os.listdir(train_path)

    os.chdir(train_path)

    for tar in tqdm(tar_list):
        dir_name = tar.split('.')[0]

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        if os.path.isdir(tar):
            continue

        with tarfile.open(tar) as t:
            for tarinfo in t:
                tarinfo.name= os.path.basename(tarinfo.name)
                t.extract(tarinfo, dir_name)

        os.remove(tar)


def valprep(path):
    os.chdir(os.path.join(path, "val"))
    os.system("./../valprep.sh")


def print_dataset_info(path):
    for item in os.listdir(path):
        print(f" - {item}")

        item_path = os.path.join(path, item)

        if not os.path.isdir(item_path):
            continue

        print(f"  + NUM CLASSES - {len(os.listdir(item_path))}")

        num_total_data = 0

        for data_path in os.listdir(item_path):
            num_total_data += len(os.listdir(os.path.join(item_path, data_path)))

        print(f"  + NUM DATA - {num_total_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet')

    args = parser.parse_args()

    # trainprep(args.data_path)
    # valprep(args.data_path)

    print_dataset_info(args.data_path)
