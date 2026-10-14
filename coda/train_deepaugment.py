"""
python train_augment.py /home/datasets/ILSVRC12

* 0404 add frequency filtering for LFC training
"""

import argparse
import os
import json
import random
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from modules.get_model import get_model
from modules.frequency import set_mask, fft3channels_batch


parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('root', metavar='DIR',
                    help='path to root of dataset')
parser.add_argument('--model_arch', default='q_preact_resnet50', type=str,
                    help='model architecture: (default: q_preact_resnet50)')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# Frequency filtering
parser.add_argument('--freq', default=None, type=str,
                    help='frequency filtering type (default: None)')
# Quant params
parser.add_argument('--q_bits_w', default=None, type=int,
                    help='weight quantization bits (default: None)')
parser.add_argument('--q_bits_a', default=None, type=int,   
                    help='activation quantization bits (default: None)')
parser.add_argument('--q_scheme', default=None, type=str,
                    help='quantization scheme (default: None)')
# save
parser.add_argument('--save_path', default='./checkpoint/deepaug/imagenet_resnet50_fp/r0r0', type=str,
                    help='path to save checkpoint (default: ./checkpoint)')
args = parser.parse_args()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


best_acc1 = 0

def get_checkpoint_path(model_arch, q_scheme=None, freq=None, q_bits=None, config_file="ckpt_paths.json"):
    """
    간단한 checkpoint 경로 가져오기 함수
    
    Args:
        model_arch (str): 모델 아키텍처
        q_scheme (str): 양자화 방식 (기본값: 'fp')
        freq (str): 주파수 필터링 ('low'면 'r0r56', 그 외 'r0r0')
        q_bits (int): 양자화 비트 (기본값: 32)
        config_file (str): JSON 설정 파일 경로
    
    Returns:
        str: checkpoint 파일 경로
    """
    # 기본값 설정
    q_scheme = 'fp' if q_scheme is None else q_scheme
    freq = 'r0r56' if freq == 'low' else 'r0r0'
    q_bits = 32 if q_bits is None else q_bits
    
    # JSON 파일 로드
    try:
        with open(config_file, 'r') as f:
            checkpoints = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint config file {config_file} not found!")
        return None
    
    # 경로 가져오기
    try:
        path = checkpoints[model_arch][q_scheme][freq][str(q_bits)]
        print(f"[INFO] Checkpoint path: {path}")
        return path
    except KeyError as e:
        print(f"[ERROR] Checkpoint not found for {model_arch}, {q_scheme}, {freq}, {q_bits}bit")
        return None

def main():
    args = parser.parse_args()
    print(f"[INFO] Arguments: {args}")

    if (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
        print(f"[INFO] Makedirs {args.save_path}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    print(f"[STATUS] Start setting model: {args.model_arch}")
    num_classes = 1000
    args.dataset = 'imagenet'
    args.pretrained = False
    model = get_model(args, num_classes)
    
    # load saved model
    checkpoint_path = get_checkpoint_path(
        model_arch=args.model_arch,
        q_scheme=args.q_scheme,
        freq=args.freq,
        q_bits=args.q_bits_w
    )
          
    ckpt = torch.load(checkpoint_path)
    if 'state_dict' in ckpt.keys(): model.load_state_dict(ckpt['state_dict'])
    else: model.load_state_dict(ckpt['model'])
    print(f"[STATUS] Load checkpoint from {checkpoint_path}")

    torch.to.set_device(args.gpu)
    model = model.to(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay, 
                               nesterov=True)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.root, 'ILSVRC12', 'train')
    valdir = os.path.join(args.root, 'ILSVRC12', 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    edsr_dataset = datasets.ImageFolder(
        '/ssd/datasets/EDSR',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    cae_dataset = datasets.ImageFolder(
        '/ssd/datasets/CAE',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, edsr_dataset, cae_dataset])

    # Frequency filtering
    if args.freq == 'low':
        low_r = 0
        high_r = 56
        args.freq_mask = set_mask(224, low_r, high_r).to(args.gpu)
    else:
        args.freq_mask = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model_arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.save_path, str(args.q_bits_w) + 'bit_checkpoint.pt'))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(args.gpu, non_blocking=True)
        target = target.to(args.gpu, non_blocking=True)

        if args.freq is not None:
            images, _ = fft3channels_batch(images, images.size(0), 3, args.freq_mask, args)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.to(args.gpu, non_blocking=True)
            target = target.to(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='./checkpoint.pt'):
    # torch.save(state, filename)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, './model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))


        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
