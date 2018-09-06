import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchnet.meter import ClassErrorMeter
from wrn_mcdonnell import WRN_McDonnell
from main import create_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Binary Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--width', default=1, type=float)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dataroot', default='.', type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    have_cuda = torch.cuda.is_available()
    def cast(x):
        return x.cuda() if have_cuda else x

    checkpoint = torch.load(args.checkpoint)

    weights_unpacked = {}
    for k, w in checkpoint.items():
        if w.dtype == torch.uint8:
            # weights are packed with np.packbits function
            scale = np.sqrt(2 / (w.shape[1] * w.shape[2] * w.shape[3] * 8))
            signed = np.unpackbits(w, axis=1).astype(np.int) * 2 - 1
            weights_unpacked[k[7:]] = torch.from_numpy(signed).float() * scale
        else:
            weights_unpacked[k[7:]] = w

    model = WRN_McDonnell(args.depth, args.width, num_classes)
    model.load_state_dict(weights_unpacked)
    model = cast(model)
    model.eval()

    class_acc = ClassErrorMeter(accuracy=True)

    for inputs, targets in tqdm(DataLoader(create_dataset(args, train=False), 256)):
        with torch.no_grad():
            class_acc.add(model(cast(inputs)).cpu(), targets)

    print(class_acc.value())


if __name__ == '__main__':
    main()
