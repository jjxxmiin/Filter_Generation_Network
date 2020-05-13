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
                 criterion,
                 train_loader=None,
                 test_loader=None,
                 optimizer=None,
                 scheduler=None,
                 device='cuda'):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.e = 0

    def train(self):
        self.model.train()

        train_iter = len(self.train_loader)

        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        for i, (images, labels) in tqdm(enumerate(self.train_loader), total=train_iter):
            if self.scheduler is not None:
                self.scheduler(self.optimizer, i, self.e)

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

        self.e += 1

        return train_loss, train_correct

    def test(self):
        self.model.eval()

        test_iter = len(self.test_loader)

        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        for i, (images, labels) in tqdm(enumerate(self.test_loader), total=test_iter):
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