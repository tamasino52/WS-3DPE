# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds

from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        inputs, targets, target_weights, metas, limb = data
        # measure data loading time
        data_time.update(time.time() - end)

        # load on cuda
        targets = [target.cuda(non_blocking=True) for target in targets]
        target_weights = [target_weight.cuda(non_blocking=True) for target_weight in target_weights]
        limb = limb.cuda(non_blocking=True)

        output_heatmaps = []
        output_depthmaps = []

        for v, (input, target, target_weight, meta) in enumerate(zip(inputs, targets, target_weights, metas)):
            output_heatmap, output_depthmap = model(input)
            output_heatmaps.append(output_heatmap)
            output_depthmaps.append(output_depthmap)

        loss = criterion(output_heatmaps, output_depthmaps, targets, target_weights, limb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), len(inputs) * inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            _, avg_acc, cnt, pred = accuracy(output_heatmap.detach().cpu().numpy(), target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * 4, output_heatmap, output_depthmap, prefix)

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    speed=inputs[0].size(0) * len(inputs) / batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        if i == 20:
            break


def validate(config, val_loader, val_dataset, model, criterion, output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.empty((0, config.NETWORK.NUM_JOINTS, 3), dtype=np.float)
    all_boxes = np.empty((0, 6), dtype=np.float)
    #all_preds = np.zeros(
    #    (num_samples, config.NETWORK.NUM_JOINTS, 3),
    #    dtype=np.float32
    #)
    #all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets, target_weights, metas, limb = data

            # load on cuda
            targets = [target.cuda(non_blocking=True) for target in targets]
            target_weights = [target_weight.cuda(non_blocking=True) for target_weight in target_weights]
            limb = limb.cuda(non_blocking=True)

            output_heatmaps = []
            output_depthmaps = []

            # compute output
            for v, (input, target, target_weight, meta) in enumerate(zip(inputs, targets, target_weights, metas)):
                output_heatmap, output_depthmap = model(input)
                output_heatmaps.append(output_heatmap)
                output_depthmaps.append(output_depthmap)

            loss = criterion(output_heatmaps, output_depthmaps, targets, target_weights, limb)

            num_images = inputs[0].size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images * (v+1))

            for output_heatmap, output_depthmap, target, meta in zip(output_heatmaps, output_depthmaps, targets, metas):

                _, avg_acc, cnt, pred = accuracy(output_heatmap.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    config, output_heatmap.clone().cpu().numpy(), c, s)
                all_preds = np.append(all_preds, np.concatenate((preds, maxvals), axis=2), axis=0)
                all_boxes = np.append(all_boxes, np.concatenate((
                    c[:, 0:2],
                    s[:, 0:2],
                    np.expand_dims(np.prod(s*200, 1), axis=1),
                    np.expand_dims(score, axis=1)
                ), axis=1), axis=0)

                '''
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                '''

                image_path.extend(meta['image'])

                #idx += num_images

            if i is 20:
                break
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output_heatmap, output_depthmap, prefix)

        all_preds = np.array(all_preds)
        all_boxes = np.array(all_boxes)
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames
        )

        model_name = config.MODEL
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
