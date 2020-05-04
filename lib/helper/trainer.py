import torch
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassifyTrainer:
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device='cuda'):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_iter = len(train_loader)
        self.test_iter = len(test_loader)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, epoch):
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        self.model.train()

        for i, (images, labels) in tqdm(enumerate(self.train_loader), total=self.train_iter):
            if self.scheduler is not None:
                self.scheduler(self.optimizer, i, epoch)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            # forward
            pred = self.model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            correct_meter.update((predicted == labels).sum().item())
            # loss
            loss = self.criterion(pred, labels)
            loss_meter.update(loss.item())
            # backward
            loss.backward()
            # weight update
            self.optimizer.step()

        train_loss = loss_meter.avg
        train_correct = correct_meter.avg

        return train_loss, train_correct

    def test(self):
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        self.model.eval()

        for i, (images, labels) in tqdm(enumerate(self.test_loader), total=self.test_iter):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # forward
            pred = self.model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            correct_meter.update((predicted == labels).sum().item())
            # loss
            loss = self.criterion(pred, labels)
            loss_meter.update(loss.item())

        test_loss = loss_meter.avg
        test_acc = correct_meter.avg

        return test_loss, test_acc

    def save(self, save_path):
        print("model saved")
        torch.save(self.model.state_dict(), save_path)