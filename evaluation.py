import time
import torch
import argparse
import matplotlib.pyplot as plt

from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import FVGG
from lib.models.module import GFLayer
from lib.helper import ClassifyTrainer
from lib.interpretable import GradCAM

plt.interactive(False)


def print_inference_time(model, args):
    total = 0

    for m in model.modules():
        if isinstance(m, GFLayer):
            total += m.weights.data.numel()
        elif isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    criterion = nn.CrossEntropyLoss().to(args.device)

    total_time = 0

    for i in range(0, 5):
        trainer = ClassifyTrainer(model,
                                  criterion,
                                  test_loader=test_loader)

        start = time.time()
        test_loss, test_acc = trainer.test()
        end = time.time()

        test_acc = test_acc / args.batch_size

        total_time += end - start

    print(f'Total Params : {total} \n'
          f' + NUM Filter : {args.num_filters} \n'
          f' + EDGE Filter : {args.edge_filter_type} \n'
          f' + TEXTURE Filter : {args.texture_filter_type} \n'
          f' + OBJECT Filter : {args.object_filter_type} \n'
          f' + Total Time : {total_time / 5} \n'
          f' + TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ] \n')


def show_grad_cam(model, label, test_loader):
    grad_cam = GradCAM(model, label)

    grad_cam.save_img(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10')

    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument('--model_name', type=str, default='vgg16')
    parser.add_argument('--datasets', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_filters', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--edge_filter_type', type=str, default='conv')
    parser.add_argument('--texture_filter_type', type=str, default='conv')
    parser.add_argument('--object_filter_type', type=str, default='conv')
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--log_path', type=str, default='./cifar10.log')

    args = parser.parse_args()

    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    name = f'{args.datasets}_' \
           f'{args.model_name}_' \
           f'{args.num_filters}_' \
           f'{args.edge_filter_type}_' \
           f'{args.texture_filter_type}_' \
           f'{args.object_filter_type}'

    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    filter_types = [args.edge_filter_type,
                    args.texture_filter_type,
                    args.object_filter_type]

    model = FVGG('VGG16',
                 num_filters=args.num_filters,
                 filter_types=filter_types).to(args.device)

    model.load_state_dict(torch.load(f'./checkpoint/{name}_model.pth'))

    # print_inference_time(model, args)
    show_grad_cam(model, label, test_loader)