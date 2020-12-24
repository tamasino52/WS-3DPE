from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle
import argparse
import numpy as np
import shutil

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from args import get_parameter

import lib
import dataset
import models
import utils

from lib.utils.utils import get_optimizer, save_checkpoint, load_checkpoint
from lib.core.config import config, update_config, update_dir

from lib.models.pose_hrnet import get_pose_net
from lib.models.multiview_pose_hrnet import get_multiview_pose_net
from lib.core.loss import JointsMSELoss, LimbLengthLoss, MultiViewConsistencyLoss
from lib.core.function import validate

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

    # CuDNN
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # HRNet Model
    pose_hrnet = get_pose_net(config, is_train=True)
    pose_hrnet.load_state_dict(torch.load(config.NETWORK.PRETRAINED), strict=False)
    mv_hrnet = get_multiview_pose_net(pose_hrnet, config)
    depth_hrnet = get_pose_net(config, is_train=True)
    print(mv_hrnet)

    # Multi GPUs Setting
    gpus = [int(i) for i in config.GPUS.split(',')]
    mv_hrnet = torch.nn.DataParallel(mv_hrnet, device_ids=gpus).cuda()

    # Loss
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    # Optimizer
    optimizer = get_optimizer(config, mv_hrnet)

    # Loading Checkpoint
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, mv_hrnet, optimizer = load_checkpoint(mv_hrnet, optimizer, final_output_dir)

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    from lib.dataset.multiview_h36m import MultiViewH36M



if __name__ == '__main__':
    main()

