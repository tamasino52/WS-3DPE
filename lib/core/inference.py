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
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.transforms import transform_preds
from lib.utils.procrustes import batch_compute_similarity_transform_torch
from lib.utils.soft_argmax import SoftArgmax1D, SoftArgmax2D


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be .transforms4-ndim'

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


class PoseReconstructor(nn.Module):
    def __init__(self):
        super(PoseReconstructor, self).__init__()
        self.soft_argmax = SoftArgmax2D(window_fn='Parzen')
        self.pdist = nn.PairwiseDistance(p=2, keepdim=True)
        self.p_key = 0
        self.c_key = 9

    def infer(self, hm, dm, cam):
        kps_25d = self.get_kps_25d(hm, dm)
        s = self.get_scaling_factor(kps_25d)
        kps_25d_hat = self.get_scale_normalized_pose(kps_25d, s)
        return kps_25d_hat
        #kps_3d_hat = self.reconstruct_3d_kps(kps_25d_hat, cam)
        #return kps_3d_hat

    def get_kps_25d(self, batch_heatmap, batch_depthmap):
        for h, d in zip(batch_heatmap.shape, batch_depthmap.shape):
            assert h is d, 'Heatmap and Depthmap have different shapes.'
        b, n, h, w = batch_heatmap.shape
        batch_soft_heatmap = torch.nn.functional.softmax(batch_heatmap.view(b, n, -1), dim=2).view(b, n, h, w)
        batch_soft_depthmap = batch_soft_heatmap.matmul(batch_depthmap)
        depth = batch_soft_depthmap.sum(dim=[2, 3]).unsqueeze(2)
        heatmap_index = self.soft_argmax(batch_heatmap)
        return torch.cat([heatmap_index, depth], dim=2)

    def get_scaling_factor(self, kps_25d):
        """
        We use the mean distance between joints to calculate the scaling factor s.
        Args:
            kps_25d:
                Batch 2.5D KeyPoints (Shape = [Batch, Joint, 3])
        Returns:
            Batch Scaling Factor s (Shape = [Batch, 1])
        """
        return self.pdist(kps_25d[:, self.p_key, :], kps_25d[:, self.c_key, :])

    def get_scale_normalized_pose(self, kps_25d, s):
        """
        Args:
            kps_25d:
                Batch 2.5D KeyPoints (Shape = [Batch, Joint, 3])
            s:
                Scale Normalization Factor
        Returns:
            Batch Scale Normalized 2.5D Pose (Shape = [Batch, Joint, 3])
        """
        if s is not 0.0:
            kps_25d_hat = kps_25d / s.unsqueeze(2)
        else:
            kps_25d_hat = kps_25d
        return kps_25d_hat

    def get_z_root(self, kps_25d_hat):
        x1 = kps_25d_hat[:, self.p_key, 0]
        y1 = kps_25d_hat[:, self.p_key, 1]
        z1 = kps_25d_hat[:, self.p_key, 2]

        x2 = kps_25d_hat[:, self.c_key, 0]
        y2 = kps_25d_hat[:, self.c_key, 1]
        z2 = kps_25d_hat[:, self.c_key, 2]

        a = (x1 - x2) ** 2 + (y1 - y2) ** 2
        b = z1 * (x1 ** 2 + y1 ** 2 - x1 * x2 - y1 * y2) + z2 * (x2 ** 2 + y2 ** 2 - x1 * x2 - y1 * y2)
        c = (x1 * z1 - x2 * z2) ** 2 + (y1 * z1 - y2 * z2) ** 2 + (z1 - z2) ** 2 - 1

        discriminant = b ** 2 - 4 * a * c
        discriminant[discriminant < 0] = (b ** 2 + 4 * a * c)[discriminant < 0]
        z_root = 0.5 * (discriminant ** 0.5) / a

        return z_root.view(-1, 1)

    def reconstruct_3d_kps(self, _kps_25d_hat, intrinsic_k):
        """
        Args:
            _kps_25d_hat:
                Batch Scale Normalized 2.5D KeyPoints (Shape = [Batch, Joint, 3] // X, Y, Relative Z)
            intrinsic_k:
                Intrinsic Camera Matrix Batch * [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        Returns:
            Batch 3D KeyPoints (Shape = [Batch, Joint, 3])
        """
        z_root = self.get_z_root(_kps_25d_hat)
        kps_25d_hat = _kps_25d_hat.clone()
        K_inv = intrinsic_k.inverse()
        K_inv = K_inv.repeat(20, 1, 1)
        kps_25d_hat[:, :, 2] = 1
        kps_3d_hat = (z_root + kps_25d_hat[:, :, 2]).view(-1, 1, 1) * K_inv.bmm(kps_25d_hat.view(-1, 3, 1))

        return kps_3d_hat.view(-1, 20, 3)

    def procrustes_transform(self, kps_3d_hat, gt_kps_3d_hat):
        """
        Args:
            kps_3d_hat:
                Batch Scale Normalized 3D KeyPoints (Shape = [Batch, Joint, 3]
            gt_kps_3d_hat:
                Batch Scale Normalized 3D KeyPoints (Shape = [Batch, Joint, 3]
        Returns:
            Batch 3D KeyPoints Aligned to GT  (Shape = [Batch, Joint, 3])
        """
        trans_kps_3d_hat = batch_compute_similarity_transform_torch(kps_3d_hat, gt_kps_3d_hat)
        return trans_kps_3d_hat