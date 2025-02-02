# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from sklearn.metrics import average_precision_score


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # uncomment the 2 lines below for single label cases
#        if mixup_fn is not None:
 #            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()


        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, timestamps, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            outputs = model(samples, timestamps)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        print(loss)

        torch.cuda.synchronize()
        # torch.no_grad()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# mAP calculator >> added for multilabel scoring
def mean_average_precision(targets, outputs):
    """
    Calculates mean average precision (mAP) for multi-label classification.

    Args:
        targets: A tensor of one-hot encoded true class labels (multiple 1s allowed).
        outputs: A tensor of predicted probabilities for each class.

    Returns:
        The mean average precision (mAP) across all classes.
    """

    with torch.no_grad():
        num_classes = targets.shape[1]
        average_precisions = []

        for class_idx in range(num_classes):
            # Get relevant indices for the current class
            relevant_idxs = (targets[:, class_idx] == 1)

            # Calculate TP, FP, and FN considering only relevant examples
            tp = (outputs[relevant_idxs, class_idx] > 0.5).sum().float()  # Threshold for positive prediction
            fp = (outputs[~relevant_idxs, class_idx] > 0.5).sum().float()
            fn = (outputs[relevant_idxs, class_idx] <= 0.5).sum().float()

            # Sort predictions by confidence (descending), but only for relevant examples
            _, indices = torch.sort(outputs[relevant_idxs, class_idx], descending=True)

            # Calculate precision and recall at each threshold
            precisions = torch.zeros_like(outputs[relevant_idxs, class_idx])
            recalls = torch.zeros_like(outputs[relevant_idxs, class_idx])
            accumulated_tp = 0

            for i, idx in enumerate(indices):
                accumulated_tp += 1
                precisions[i] = accumulated_tp / (i + 1)
                recalls[i] = accumulated_tp / (tp + fn)

            # Calculate average precision for the class
            average_precision = torch.trapz(precisions, recalls)
            average_precisions.append(average_precision)

        # Calculate mean average precision across all classes
        mAP = torch.mean(torch.tensor(average_precisions))

    return mAP



@torch.no_grad()
def evaluate(data_loader, model, device):
    # needs to be adapted for both single and multilabel
    # single label commented out >> crossentropy
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()


    # all changes from here and below are related to metrics/acuracy and metric logger
    # for single label vs multilabel
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)


#        acc1, acc5 = accuracy(output, target, topk=(1, 5))


        mpa = mean_average_precision(target, output)
        #print("mAP", mpa)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mAP'].update(mpa.item(), n=batch_size)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(mAP=metric_logger.meters['mAP'], losses=metric_logger.loss))

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(data_loader, model, device):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    tta = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        timestamps = batch[1]
        target = batch[-1]

        batch_size = images.shape[0]
        # print(images.shape, timestamps.shape, target.shape)
        if tta:
            images = images.reshape(-1, 3, 3, 224, 224)
            timestamps = timestamps.reshape(-1, 3, 3)
            target = target.reshape(-1, 1)
        # images = images.reshape()
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
        # with torch.no_grad():
            output = model(images, timestamps)

            if tta:
                # output = output.reshape(batch_size, 9, -1).mean(dim=1, keepdims=False)

                output = output.reshape(batch_size, 9, -1)
                sp = output.shape
                maxarg = output.argmax(dim=-1)

                output = F.one_hot(maxarg.reshape(-1), num_classes=1000).float()
                output = output.reshape(sp).mean(dim=1, keepdims=False)
                
                target = target.reshape(batch_size, 9)[:, 0]

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}