import torch
import argparse
import matplotlib.pyplot as plt

from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import FVGG, GFLayer
from lib.helper import ClassifyTrainer

plt.interactive(False)

parser = argparse.ArgumentParser(description='CIFAR10')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--percent', type=float, default=0.1)
parser.add_argument('--filter_type', type=str, default='uniform')
parser.add_argument('--save_path', type=str, default='./checkpoint')

args = parser.parse_args()

ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
origin_acc = []
origin_loss = []
prune_acc = []
prune_loss = []

for r in ratio:
    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    model = FVGG('VGG11',
                 num_filters=args.num_filters,
                 filter_type=args.filter_type,
                 stride=args.stride).to(args.device)

    model.load_state_dict(torch.load(f'./{args.save_path}/cifar10_{args.num_filters}.pth'))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trainer = ClassifyTrainer(model,
                              criterion,
                              test_loader=test_loader)

    test_loss, test_acc = trainer.test()
    test_acc = test_acc / args.batch_size

    print(f" + ORIGIN TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")
    origin_acc.append(test_acc)
    origin_loss.append(test_loss)

    total = 0

    for m in model.modules():
        if isinstance(m, GFLayer):
            total += m.weights.data.numel()

    conv_weights = torch.zeros(total).cuda()
    index = 0

    for m in model.modules():
        if isinstance(m, GFLayer):
            size = m.weights.data.numel()
            conv_weights[index:(index + size)] = m.weights.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * r)
    thre = y[thre_index]

    pruned = 0
    print(f'Pruning Threshold : {thre}')
    zero_flag = False

    for k, m in enumerate(model.modules()):
        if isinstance(m, GFLayer):
            weight_copy = m.weights.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weights.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print(f'layer index: {k} \t total params: {mask.numel()} \t remaining params: {int(torch.sum(mask))}')

    prune_trainer = ClassifyTrainer(model,
                                    criterion,
                                    test_loader=test_loader)

    test_loss, test_acc = prune_trainer.test()
    test_acc = test_acc / args.batch_size

    print(f" + PRUNE TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")

    prune_acc.append(test_acc)
    prune_loss.append(test_loss)

    print(f'Total conv params: {total}, Pruned conv params: {pruned}, Pruned ratio: {pruned / total}')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].plot(origin_acc, label='origin acc')
axes[0, 0].plot(prune_acc, label='prune acc')
axes[0, 0].set_title("ACC")
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].plot(origin_loss, label='origin loss')
axes[0, 1].plot(prune_loss, label='prune loss')
axes[0, 1].set_title("Loss")
axes[0, 1].legend()
axes[0, 1].grid()

plt.show()

