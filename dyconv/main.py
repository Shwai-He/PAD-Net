from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from model import model_entry
from utils.misc import parse_config, get_temperature
import torchvision.models as models
import models.imagenet as customized_models
from utils import Logger, AverageMeter, accuracy, mkdir_p
from utils.imagenet import ImageNet 
from tensorboardX import SummaryWriter
from prune import _pruner, prune_loop
from pruners import *
from utils import generator
from math import cos, pi
from torch import nn

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', '--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--fc-squeeze', type=int, default=8, help='squeeze of fc.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup or not')
parser.add_argument('--mode', type=str, default='large', help='large or small MobileNetV3')
parser.add_argument('--dropout', type=float, default=None, help='drop out ratio.')
parser.add_argument('--device_ids', type=str, default="4")
parser.add_argument('--config_file', type=str, default="")
parser.add_argument('--sparsity', type=float, default=0.5)
parser.add_argument('--r', action='store_true')
parser.add_argument('--prune_dataset_ratio', type=int, default=10,
                         help='ratio of prune dataset size and number of classes (default: 10)')
parser.add_argument('--pruner', type=str, default='rand',
                         choices=['rand', 'mag', 'snip', 'grasp', 'synflow', 'varient'],
                         help='prune strategy (default: rand)')
parser.add_argument('--prune-bias', type=bool, default=False,
                                  help='whether to prune bias parameters (default: False)')
parser.add_argument('--prune-batchnorm', type=bool, default=False,
                         help='whether to prune batchnorm layers (default: False)')
parser.add_argument('--prune-residual', type=bool, default=False,
                         help='whether to prune residual connections (default: False)')
parser.add_argument('--prune-train-mode', type=bool, default=False,
                         help='whether to prune in train mode (default: False)')
parser.add_argument('--reinitialize', type=bool, default=False,
                         help='whether to reinitialize weight parameters after pruning (default: False)')
parser.add_argument('--shuffle', type=bool, default=False,
                         help='whether to shuffle masks after pruning (default: False)')
parser.add_argument('--invert', type=bool, default=False,
                         help='whether to invert scores during pruning (default: False)')
parser.add_argument('--pruner-list', type=str, nargs='*', default=[],
                         help='list of pruning strategies for singleshot (default: [])')
parser.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                         help='list of prune epochs for singleshot (default: [])')
parser.add_argument('--compression-list', type=float, nargs='*', default=[],
                         help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--level-list', type=int, nargs='*', default=[],
                         help='list of number of prune-train cycles (levels) for multishot (default: [])')
parser.add_argument('--compression-schedule', type=str, default='exponential',
                         choices=['linear', 'exponential'],
                         help='whether to use a linear or exponential compression schedule (default: exponential)')
parser.add_argument('--mask-scope', type=str, default='global', choices=['global', 'local'],
                                  help='masking scope (global or layer) (default: global)')
parser.add_argument('--prune-epochs', type=int, default=1,
                         help='number of iterations for scoring (default: 1)')
parser.add_argument('--temp_epoch', type=int, default=15)
parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')

best_prec1 = 0
Pearson = []
def print_options(save_path, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    # save to the disk
    file_name = os.path.join(save_path, 'options.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    global args, best_prec1, step_note, device
    args = parser.parse_args()
    config = parse_config(args.config_file)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    set_random_seeds(random_seed=0)
    device_ids = [int(device_id) for device_id in args.device_ids.split(' ')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids.replace(' ', ',')
    # device = torch.device("cuda:{}".format(device_ids[0]))
    device = torch.device("cuda:0")
    args.rank = args.world_size = len(device_ids)
    args.distributed = args.world_size > 1
    args.distributed = False
    # torch.cuda.set_device(device_ids[0])
    print("=> creating model '{}'".format(args.arch))
    model = model_entry(config.model)
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            # model.cuda()
            model.to(device)
        else:
            # model = torch.nn.DataParallel(model, device_ids=device_ids)
            model = torch.nn.DataParallel(model)
            model.to(device)

    else:
        # dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.world_size)
        torch.distributed.init_process_group(backend="nccl")
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # define loss function (criterion) and optimizer
    if args.label_smoothing > 0:
        # using Label Smoothing
        criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),  # 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageNet(
        traindir,
        train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ImageNet(valdir, transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.CenterCrop(args.input_size),  # 224
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_loader_len, val_loader_len = len(train_loader), len(val_loader)

    if args.evaluate:
        from collections import OrderedDict
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            source_state = checkpoint['state_dict']
            if 'temperature' in checkpoint:
                temperature = checkpoint['temperature']
                model.module.temperature = temperature
            else:
                model.module.temperature = 1.
            target_state = OrderedDict()
            for k, v in source_state.items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            model.load_state_dict(target_state)
        else:
            print("=> no weight found at '{}'".format(args.weight))

        validate(val_loader, val_loader_len, model, criterion)
        return
    #todo mode partition
    if 'pad' in config.model['type'] and not (args.resume and args.r):
        length = args.prune_dataset_ratio
        indices = torch.randperm(len(train_dataset))[:length * args.batch_size]
        prune_dataset = torch.utils.data.Subset(train_dataset, indices)
        prune_loader = torch.utils.data.DataLoader(
            prune_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        pruner = _pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias))
        prune_loop(model, criterion, pruner, prune_loader, device, args.sparsity,
                   args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
        model.module.reset_buffers()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    arch = open(os.path.join(args.checkpoint, 'arch.txt'), 'w')

    print(model, file=arch)
    arch.close()

    if os.path.exists(os.path.join(args.checkpoint, 'checkpoint.pth.tar')):
        args.resume = os.path.join(args.checkpoint, 'checkpoint.pth.tar')
    global logger
    if args.resume and args.r:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            temperature = checkpoint['temperature']
            model.load_state_dict(checkpoint['state_dict'])
            if 'pad' in config.model['type']:
                model.module.reset_buffers()
            model.module.temperature = temperature
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            total, nonzero = 0, 0
            if 'pad' in config.model['type']:
                for name, buf in model.module.named_buffers():
                    if 'kernel' in name:
                        total += buf.numel()
                        nonzero += buf.mean() * buf.numel()
                        print(buf.numel(), buf.mean())
                print(nonzero / total)
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid Acc5.'])
        print_options(args.checkpoint, args)



    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        # # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1, prec5 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([epoch+1, lr, train_loss, val_loss, train_acc, prec1, prec5])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'temperature': model.module.temperature,
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)


    logger.close()
    writer.close()


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    if epoch < 100:
        mixup_alpha = args.mixup * float(epoch) / 100
    else:
        mixup_alpha = args.mixup
    # for i in range(len(train_loader)):
    for i, (input, target) in enumerate(train_loader):
        if epoch < args.temp_epoch and hasattr(model.module, 'update_temperature'):
            temp = get_temperature(i, epoch, train_loader_len,
                                   temp_epoch=args.temp_epoch, temp_init=args.temp_init)
            model.module.update_temperature(temp)

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        input, target = torch.autograd.Variable(input), torch.autograd.Variable(target)
        # compute output
        if args.mixup != 0:
            # using mixup
            input, label_a, label_b, lam = mixup_data(input, target, mixup_alpha)
            output = model(input)
            loss = mixup_criterion(criterion, output, label_a, label_b, lam)
            acc1_a, acc5_a = accuracy(output, label_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(output, label_b, topk=(1, 5))
            # measure accuracy and record loss
            prec1 = lam * acc1_a + (1 - lam) * acc1_b
            prec5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            # normal forward
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 500 == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, args.epochs, i, train_loader_len, optimizer.param_groups[0]['lr'], batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)

def validate(val_loader, val_loader_len, model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        # input = input.cuda()
        # target = target.cuda()

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']
    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mixup_data(x, y, alpha):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = input.log_softmax(dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



if __name__ == '__main__':
    main()
