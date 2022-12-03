import math
import time

import torch

from lib.utils.utils import print_speed


def train(train_loader, model, optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        template = input[0].to(device, non_blocking=True)
        search = input[1].to(device, non_blocking=True)
        label_cls = input[2].float().to(device, non_blocking=True)  # BCE need float
        reg_label = input[3].float().to(device, non_blocking=True)
        reg_weight = input[4].float().to(device, non_blocking=True)
        del input

        cls_loss_ori, cls_loss_align, reg_loss = model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + reg_loss  # smaller reg loss is better for stable training (compared to 1.2 in SiamRPN seriese)
        else:  # I would suggest the readers to perform ablation on the loss trade-off weights when building a new module
            cls_loss_align = 0
            loss = cls_loss_ori + reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()

        loss.backward()

        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}  Batch Time: {batch_time.avg:.3f}s  Data Time:{data_time.avg:.3f}s'
                ' \n CLS_ORI Loss:{cls_loss_ori.avg:.5f}  CLS_ALIGN Loss:{cls_loss_align.avg:.5f}'
                '  REG Loss:{reg_loss.avg:.5f}  Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


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
        self.avg = self.sum / self.count if self.count != 0 else 0
