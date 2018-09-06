"""Implementation of 1-bit Wide Residual Networks

Implements ICLR 2018 paper:
"Training wide residual networks for deployment using a single bit for each weight"
by Mark D. McDonnell

2018 Sergey Zagoruyko
"""
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.nn import DataParallel
from torch.backends import cudnn
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchnet as tnt
from wrn_mcdonnell import WRN_McDonnell

cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--width', default=1, type=float)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dataroot', default='.', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--binarize', action='store_true')

    # Training options
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr-min', default=0.0001, type=float)
    parser.add_argument('--epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--restarts', default='[2,4,8,16,32,64,128]', type=json.loads,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default='', type=str,
                        help='save parameters and logs in this folder')
    return parser.parse_args()


def create_dataset(args, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, args.dataset)(args.dataroot, train=train, download=True, transform=transform)


def main():
    args = parse_args()
    print('parsed options:', vars(args))

    have_cuda = torch.cuda.is_available()
    def cast(x):
        return x.cuda() if have_cuda else x

    torch.manual_seed(args.seed)

    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    def create_iterator(mode):
        return DataLoader(create_dataset(args, mode), args.batch_size, shuffle=mode,
                          num_workers=args.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    model = WRN_McDonnell(args.depth, args.width, num_classes, args.binarize)
    model = cast(DataParallel(model))

    n_parameters = sum(p.numel() for p in model.parameters())

    optimizer = SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=args.lr_min)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    if not Path(args.save).exists():
        Path(args.save).mkdir()

    def log(log_data):
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': log_data['epoch'],
                   }, Path(args.save) / 'checkpoint.pth.tar')
        z = {**vars(args), **log_data}
        with open(Path(args.save) / 'log.txt', 'a') as f:
            f.write(json.dumps(z) + '\n')
        print(z)

    def train():
        model.train()
        meter_loss = tnt.meter.AverageValueMeter()
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)
        train_iterator = tqdm(train_loader, dynamic_ncols=True)
        for x, y in train_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = cross_entropy(outputs, cast(y))
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            train_iterator.set_postfix(loss=loss.item())
            classacc.add(outputs.data.cpu(), y.cpu())
        return meter_loss.mean, classacc.value()[0]

    def test():
        model.eval()
        meter_loss = tnt.meter.AverageValueMeter()
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)
        test_iterator = tqdm(test_loader, dynamic_ncols=True)
        for x, y in test_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = cross_entropy(outputs, cast(y))
            loss.backward()
            meter_loss.add(loss.item())
            classacc.add(outputs.data.cpu(), y.cpu())
        return meter_loss.mean, classacc.value()[0]

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        if epoch in args.restarts:
            scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.lr_min)
        train_loss, train_acc = train()
        test_loss, test_acc = test()
        log_data = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch": epoch,
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "lr": scheduler.get_lr(),
        }
        log(log_data)
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
              (args.save, epoch, args.epochs, test_acc))


if __name__ == '__main__':
    main()
