# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_scaling_factor(kps_25d):
    """
    We use the mean distance between joints to calculate the scaling factor s.
    Args:
        kps_25d:
            Batch 2.5D KeyPoints (Shape = [Batch, Joint, X, Y, Relative Z])
    Returns:
        Batch Scaling Factor s
    """
    human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                    (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
    parent = kps_25d[:, [item[0] for item in human36_edge], :]
    child = kps_25d[:, [item[1] for item in human36_edge], :]
    dist = torch.norm(parent - child, dim=2).unsqueeze(2)
    s = dist.sum([1, 2]).view(-1, 1)
    return s


def get_scale_normalized_pose(kps_25d):
    """
    Args:
        kps_25d:
            Batch 2.5D KeyPoints (Shape = [Batch, Joint, X, Y, Relative Z])
    Returns:
        Batch Scale Normalized 2.5D Pose (Shape = [Batch, Joint, X, Y, Relative Z])
    """
    s = get_scaling_factor(kps_25d)
    if s is not 0.0:
        kps_25d_hat = kps_25d / s.unsqueeze(2)
    else:
        kps_25d_hat = kps_25d
    return kps_25d_hat


def get_z_root(kps_25d_hat):
    x1 = kps_25d_hat[:, 0, 0]
    y1 = kps_25d_hat[:, 0, 1]
    z1 = kps_25d_hat[:, 0, 2]

    x2 = kps_25d_hat[:, 8, 0]
    y2 = kps_25d_hat[:, 8, 1]
    z2 = kps_25d_hat[:, 8, 2]

    a = (x1 - x2) ** 2 + (y1 - y2) ** 2
    b = z1 * (x1 ** 2 + y1 ** 2 - x1 * x2 - y1 * y2) + z2 * (x2 ** 2 + y2 ** 2 - x1 * x2 - y1 * y2)
    c = (x1 * z1 - x2 * z2) ** 2 + (y1 * z1 - y2 * z2) ** 2 + (z1 - z2) ** 2 - 1

    discriminant = b ** 2 - 4 * a * c
    discriminant[discriminant < 0] = (b ** 2 + 4 * a * c)[discriminant < 0]

    z_root = 0.5 * (discriminant ** 0.5) / a

    return z_root.view(-1, 1)


def reconstruct_3d_kps(_kps_25d_hat, intrinsic_k):
    """
    Args:
        _kps_25d_hat:
            Batch Scale Normalized 2.5D KeyPoints (Shape = [Batch, Joint, 3] // X, Y, Relative Z)
        intrinsic_k:
            Intrinsic Camera Matrix Batch * [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Returns:
        Batch 3D KeyPoints (Shape = [Batch, Joint, 3])
    """
    z_root = get_z_root(_kps_25d_hat)
    kps_25d_hat = _kps_25d_hat.clone()
    K_inv = intrinsic_k.inverse()
    K_inv = K_inv.repeat(20, 1, 1)
    kps_25d_hat[:, :, 2] = 1
    kps_3d_hat = (z_root + kps_25d_hat[:, :, 2]).view(-1, 1, 1) * K_inv.bmm(kps_25d_hat.view(-1, 3, 1))

    return kps_3d_hat.view(-1, 20, 3)
