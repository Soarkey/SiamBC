import argparse
import os
import pprint
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from lib.models.model import SiamBC
from lib.core.config import config, update_config
from lib.core.function import train
from lib.dataset.dataset import TrackingDataset
from lib.utils.utils import build_lr_scheduler, create_logger, load_pretrain, restore_from, save_model

eps = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description='Train SiamBC')
    parser.add_argument('--cfg', type=str, default='experiments/train/base.yaml', help='yaml configure file name')
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    seed_torch(seed=123456)
    args = parser.parse_args()
    return args


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'


def build_opt_lr(cfg, model, current_epoch=0):
    # fix all backbone first
    for param in model.features.features.parameters():
        param.requires_grad = False
    for m in model.features.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if current_epoch >= cfg.TRAIN.UNFIX_EPOCH:
        if len(cfg.TRAIN.TRAINABLE_LAYER) > 0:  # specific trainable layers
            for layer in cfg.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.features.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:  # train all backbone layers
            for param in model.features.features.parameters():
                param.requires_grad = True
            for m in model.features.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.features.parameters():
            param.requires_grad = False
        for m in model.features.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.features.features.parameters()),
                          'lr'    : cfg.TRAIN.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    try:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr'    : cfg.TRAIN.BASE_LR}]
    except:
        pass

    trainable_params += [{'params': model.connect_model.parameters(),
                          'lr'    : cfg.TRAIN.BASE_LR}]

    try:
        trainable_params += [{'params': model.align_head.parameters(),
                              'lr'    : cfg.TRAIN.BASE_LR}]
    except:
        pass

    # print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)

    if cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(trainable_params,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=cfg.TRAIN.ADAMW_LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.TRAIN.END_EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer'            : SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # build model
    model = SiamBC(align=config.TRAIN.ALIGN).cuda()
    print(model)

    model = load_pretrain(model, config.TRAIN.PRETRAIN)  # load pretrain
    logger.info("load pretrain model from {}".format(config.TRAIN.PRETRAIN))

    # get optimizer
    if not config.TRAIN.START_EPOCH == config.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = build_opt_lr(config, model, config.TRAIN.START_EPOCH)
    else:
        optimizer, lr_scheduler = build_opt_lr(config, model, 0)  # resume wrong (last line)
    logger.info("build optimizer and lr_scheduler")

    # check trainable again
    print('==========double check trainable==========')
    check_trainable(model, logger)  # print trainable params info

    if config.TRAIN.RESUME and config.TRAIN.START_EPOCH != 0:  # resume
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.TRAIN.RESUME)
    logger.info("restore from {}".format(config.TRAIN.RESUME))

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    # accelerate dataloader speed
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super(DataLoaderX, self).__iter__(), max_prefetch=6)

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):
        train_set = TrackingDataset(config)
        train_loader = DataLoaderX(train_set,
                                   batch_size=config.TRAIN.BATCH * gpu_num,
                                   num_workers=config.WORKERS,
                                   pin_memory=True,
                                   sampler=None,
                                   drop_last=True)

        # check if it's time to train backbone
        if epoch == config.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = build_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            check_trainable(model, logger)  # print trainable params info

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()

        model, writer_dict = train(train_loader, model, optimizer, epoch + 1, curLR, config, writer_dict, logger, device=device)

        # save model
        save_model(model, epoch, optimizer, config.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
