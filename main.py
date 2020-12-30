from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle
import argparse
import numpy as np
import pprint
import shutil

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# Local
import lib
import dataset
import models
import utils

# Utils
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import load_checkpoint
from lib.utils.utils import create_logger

# Configs
from lib.core.config import config
from lib.core.config import update_dir
from lib.core.config import get_model_name

# Functions
from lib.core.function import train
from lib.core.function import validate

# Models
from lib.core.loss import JointsMSELoss, LimbLengthLoss, MultiViewConsistencyLoss
from lib.models.pose_hrnet import get_pose_net
from lib.models.multiview_pose_hrnet import get_multiview_pose_net

# Datasets
from lib.dataset.h36m import H36MDataset

"""
from utils.utils import create_logger
from multiviews.pictorial_cuda import rpsm
from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='log')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers


def main():
    final_output_dir = 'output'
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(config, 'train')

    logger.info(pprint.pformat(config))

    # CuDNN
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # HRNet Model
    pose_hrnet = get_pose_net(config, is_train=True)  # Pose estimation model
    pose_hrnet.load_state_dict(torch.load(config.NETWORK.PRETRAINED), strict=False)  # Pretrained weight loading
    mv_hrnet = get_multiview_pose_net(pose_hrnet, config)  # Multiview adopting
    #depth_hrnet = get_pose_net(config, is_train=True)  # 2.5D depth prediction model

    # Multi GPUs Setting
    gpus = [int(i) for i in config.GPUS.split(',')]
    mv_hrnet = torch.nn.DataParallel(mv_hrnet, device_ids=gpus).cuda()
    logger.info('=> init data parallel model')

    # Loss
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    logger.info('=> init criterion')

    # Optimizer
    optimizer = get_optimizer(config, mv_hrnet)
    logger.info('=> init {} optimizer'.format(config.TRAIN.OPTIMIZER))

    # Loading checkpoint
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, mv_hrnet, optimizer = load_checkpoint(mv_hrnet, optimizer, final_output_dir)
        logger.info('=> init {} optimizer'.format(config.TRAIN.OPTIMIZER))

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    logger.info('=> init scheduler')

    # Summary
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = H36MDataset(config, config.DATASET.TRAIN_SUBSET, True,
                                transforms.Compose([transforms.ToTensor(), normalize]))
    logger.info('=> loading train dataset')

    valid_dataset = H36MDataset(config, config.DATASET.TEST_SUBSET, False,
                                transforms.Compose([transforms.ToTensor(), normalize]))
    logger.info('=> loading validation dataset')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)
    logger.info('=> init train data loader')

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    logger.info('=> init validation data loader')

    # Training loop
    best_perf = 0.0
    best_model = False
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # Trainer
        train(config, train_loader, mv_hrnet, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # Performance indicator
        perf_indicator = validate(config, valid_loader, valid_dataset, mv_hrnet,
                                  criterion, final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': mv_hrnet.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    # End
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(mv_hrnet.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

