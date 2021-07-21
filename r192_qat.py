#!/usr/bin/python3

""" load pretrained ENet_lite0 model """""" import packages """
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import sys
import shutil
import time
import logging
import argparse
from datetime import datetime

from log import setup_logging, ResultsLog, save_checkpoint
from meters import AverageMeter, accuracy
from optim import OptimRegime

parser = argparse.ArgumentParser()
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--data-dir',default = '/media/ssd0/imagenet/',
                    help='dataset dir')
parser.add_argument('--regime',default = 'sgd',
                    help='training regime')
""" Download Dataset """
def gen_loaders(path, BATCH_SIZE, NUM_WORKERS):
    # Data loading code
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(192),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(192),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    return (train_loader, val_loader)

def forward(data_loader, model, criterion, epoch, training, optimizer=None, backup_model=None):
    if training:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    total_steps=len(data_loader)

    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.to(device)
        target = target.to(device)

        if training:
            # save the model before quantized
            backup_model.load_state_dict(model.state_dict())

            # quantization effact
            model_quant(model)

        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(float(loss), inputs.size(0))
        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        if training:
            optimizer.update(epoch, epoch * len(data_loader) + i)
            optimizer.zero_grad()
            loss.backward()
            '''
            change the model weight back from quantized version
            '''
            # restore the model before applying gradient
            model.load_state_dict(backup_model.state_dict())
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if training and i % args.log_interval == 0:
            logging.info('[{0}][{1}/{2}] '
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         'Data {data_time.val:.2f} '
                         'loss {loss.val:.3f} ({loss.avg:.3f}) '
                         '@1 {top1.val:.3f} ({top1.avg:.3f}) '
                         '@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def dpu_quantize(w, BITWIDTH = 7.0):
    ''' in place 8 bits fix point quantization '''
    with torch.no_grad():
        fp_range = torch.max(torch.absolute(w))

        # no clipping
        frac_digits = BITWIDTH-torch.ceil(torch.log2(fp_range))
        # normal rounding
        int_value =  torch.round(w*2.0**frac_digits)

        # allow clipping
        # frac_digits = BITWIDTH-torch.ceil(torch.log2(0.9*fp_range))
        # int_value =  torch.round(torch.clamp(w,-0.9*fp_range,0.9*fp_range)*2.0**frac_digits)

        # stochastic rounding 1
        # int_value = torch.floor(w*2.0**frac_digits)
        # frac_value = w*2.0**frac_digits - int_value
        # int_value += (frac_value>=torch.rand(int_value.shape,device=device)).float()

        # stochastic rounding 2
        # int_value =  (w*2.0**frac_digits).type(torch.int).type(torch.float)
        # frac_value = (w*2.0**frac_digits - int_value)*torch.sign(int_value)
        # round_decision = (frac_value>=torch.rand(int_value.shape,device=device)).float()
        # int_value -= round_decision*torch.sign(w)

        # sign rounding
        # int_value = torch.floor(w*2.0**frac_digits)
        # round_decision = (torch.sign(w.grad)==1).float()
        # int_value += round_decision

        fix_point_value = int_value*1.0/(2.0**frac_digits)
        w.data = fix_point_value

def layer_quant(l):
    dpu_quantize(l.weight)
    dpu_quantize(l.bias)

def model_quant(m):
    layer_quant(m.module.conv_stem)
    for blocks in m.module.blocks:
        for b in blocks:
            if hasattr(b, 'conv_pwl'):
                layer_quant(b.conv_pwl)
            layer_quant(b.conv_dw)
            layer_quant(b.conv_pw)
    layer_quant(m.module.conv_head)
    layer_quant(m.module.classifier)

def fix_quant(w, BITWIDTH = 7.0):
    ''' get the quantization value '''
    with torch.no_grad():
        fp_range = torch.max(torch.absolute(w))
        frac_digits = BITWIDTH-torch.ceil(torch.log2(fp_range))

        # normal rounding
        int_value =  torch.round(w*2.0**frac_digits)

        # stochastic rounding
        # int_value = torch.floor(w*2.0**frac_digits)
        # frac_value = w*2.0**frac_digits - int_value
        # int_value += (frac_value>=torch.rand(int_value.shape,device=device)).float()

        fix_point_value = int_value*1.0/(2.0**frac_digits)
        return fix_point_value

def main():
    global args
    args = parser.parse_args()

    ''' set up log and training results save path '''
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = time_stamp
    save_path = os.path.join('./results', log_path)
    model_save_path = os.path.join(save_path, 'r192_quant_best.pth')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'results.log'),resume=False)
    results_path = os.path.join(save_path, 'results')
    results= ResultsLog(results_path,
                        title=' training results - %s' % log_path)

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    model = torch.load('./efn-r192-fold.pth')
    backup_model = torch.load('./efn-r192-fold.pth')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    """  Deploy model on GPU """
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device)

    backup_model = nn.DataParallel(backup_model)
    backup_model = backup_model.to(device)

    """ Optimizer & Criterion """
    sgd_regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-3,
                   'weight_decay': 5e-4 ,'momentum': 0.9},
                  {'epoch': 5, 'optimizer': 'SGD', 'lr': 1e-4}]

    adam_regime = [{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3}]

    if args.regime == 'sgd':
        regime = sgd_regime
    else:
        regime = adam_regime

    logging.info('training regime: %s', regime)
    optimizer = OptimRegime(model.parameters(), regime)

    criterion = nn.CrossEntropyLoss()

    """ Dataset"""
    train_loader, test_loader = gen_loaders(args.data_dir, args.batch_size, args.workers)
    best_prec1 = 0

    for epoch in range(0, args.epochs):
        ''' train '''
        train_loss, train_prec1, train_prec5= forward(
            train_loader, model, criterion, epoch, training=True, optimizer = optimizer, backup_model=backup_model)

        model_quant(model)

        """ test"""
        val_loss, val_prec1, val_prec5= forward(
            test_loader, model, criterion, epoch, training=False)

        is_best = val_prec1 > best_prec1
        if is_best:
            torch.save(model, model_save_path)

        best_prec1 = max(val_prec1, best_prec1)

        logging.info("best_prec1: %f %s", best_prec1,save_path)
        logging.info('Epoch: {0} '
                     'Train Prec@1 {train_prec1:.3f} '
                     'Train Prec@5 {train_prec5:.3f} '
                     'Valid Prec@1 {val_prec1:.3f} '
                     'Valid Prec@5 {val_prec5:.3f} \n'
                     .format(epoch,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch,
                    train_error1= 100 - train_prec1,
                    val_error1= 100 - val_prec1,
                    train_error5= 100 - train_prec5,
                    val_error5= 100 - val_prec5,
                   )

        results.plot(x='epoch', y=['train_error1', 'val_error1'],
                     legend=['train', 'val'],
                     title='Error@1', ylabel='error %')

        results.plot(x='epoch', y=['train_error5', 'val_error5'],
                     legend=['train', 'val'],
                     title='Error@5', ylabel='error %')

        results.save()

if __name__ == '__main__':
    main()
