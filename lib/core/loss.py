# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_batch_svd import svd
from lib.utils.vis import vis_3d_multiple_skeleton, vis_3d_skeleton
from lib.core.inference import get_max_preds
from lib.utils.soft_argmax import SoftArgmax1D, SoftArgmax2D
from lib.utils.procrustes import criterion_procrustes, compute_similarity_transform_torch




'''
def get_max_preds(batch_heatmap):
    batch_size = batch_heatmap.shape[0]
    num_joints = batch_heatmap.shape[1]
    width = batch_heatmap.shape[3]
    heatmaps_reshaped = batch_heatmap.view(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.view(batch_size, num_joints, 1)
    idx = idx.view(batch_size, num_joints, 1)

    preds = idx.repeat_interleave(2, 2).type(torch.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.gt(maxvals, 0.0).repeat_interleave(2, 2)
    pred_mask = pred_mask.type(torch.float32)

    preds *= pred_mask
    return preds, maxvals
'''


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class LimbLengthLoss(nn.Module):
    def __init__(self):
        super(LimbLengthLoss, self).__init__()
        self.human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                             (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        self.criterion = nn.MSELoss(reduction='mean')
        self.batch_norm = nn.BatchNorm1d(16)

    def get_limb_length(self, output):
        parent = torch.index_select(output, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))
        dist = torch.norm(parent-child, dim=2).unsqueeze(2)
        return dist

    def get_limb_weight(self, output_weight):
        parent = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))
        return parent.mul(child)

    def forward(self, joints_3ds):
        loss = 0
        for root_joints_3d in joints_3ds:
            root_length_pred = self.get_limb_length(root_joints_3d)
            #root_length_pred_hat = root_length_pred / root_length_pred.mean()

            for joints_3d in joints_3ds:
                length_pred = self.get_limb_length(joints_3d)
                #length_pred_hat = length_pred / length_pred.mean()
                loss += self.criterion(root_length_pred, length_pred)
        return loss


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self):
        super(MultiViewConsistencyLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                             (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]

    def forward(self, joints_3ds, target_weights):

        loss = 0
        for batch_root_joints_3d, batch_root_target_weight in zip(joints_3ds, target_weights):
            for batch_joints_3d, batch_target_weight in zip(joints_3ds, target_weights):
                for b in range(batch_root_joints_3d.shape[0]):
                    root_joints_3d = batch_root_joints_3d[b]
                    joints_3d = batch_joints_3d[b]
                    root_target_weight = batch_root_target_weight[b]
                    target_weight = batch_target_weight[b]

                    target_weight = (target_weight * root_target_weight).view(-1)

                    root_joints_3d = root_joints_3d[target_weight > 0, :]
                    joints_3d = joints_3d[target_weight > 0, :]

                    joints_3d_hat = compute_similarity_transform_torch(joints_3d, root_joints_3d)
                    loss += self.criterion(joints_3d_hat, root_joints_3d)
        return loss


class WeaklySupervisedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion1 = JointsMSELoss(use_target_weight)
        self.criterion2 = LimbLengthLoss()
        self.criterion3 = MultiViewConsistencyLoss()
        self.mse = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.alpha = 10.0
        self.beta = 100.0
        self.SoftArgmax2D = SoftArgmax2D(window_fn='Parzen')

    def get_scaling_factor(self, kps_25d):
        """
        We use the distance between the neck(0) and pelvis(8) joints to calculate the scaling factor s.
        Args:
            kps_25d:
                Batch 2.5D KeyPoints (Shape = [Batch, Joint, X, Y, Relative Z])
        Returns:
            Batch Scaling Factor s
        """
        s = ((kps_25d[:, 0, 0:2] - kps_25d[:, 8, 0:2]) ** 2).sum(1).view(-1, 1) ** 0.5
        return s

    def get_scale_normalized_pose(self, kps_25d):
        """
        Args:
            kps_25d:
                Batch 2.5D KeyPoints (Shape = [Batch, Joint, X, Y, Relative Z])
        Returns:
            Batch Scale Normalized 2.5D Pose (Shape = [Batch, Joint, X, Y, Relative Z])
        """
        s = self.get_scaling_factor(kps_25d)
        if s is not 0.0:
            kps_25d_hat = kps_25d / s.view(-1, 1, 1)
        else:
            kps_25d_hat = kps_25d
        return kps_25d_hat

    def get_z_root(self, kps_25d_hat):
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

    def get_kps_25d(self, batch_heatmap, batch_depthmap):
        for h, d in zip(batch_heatmap.shape, batch_depthmap.shape):
            assert h is d, 'Heatmap and Depthmap have different shapes.'
        b, n, h, w = batch_heatmap.shape
        batch_soft_heatmap = torch.nn.functional.softmax(batch_heatmap.view(b, n, -1), dim=2).view(b, n, h, w)
        batch_soft_depthmap = batch_soft_heatmap.matmul(batch_depthmap)
        depth = batch_soft_depthmap.sum(dim=[2, 3]).unsqueeze(2)
        heatmap_index = self.SoftArgmax2D(batch_heatmap)
        return torch.cat([heatmap_index, depth], dim=2)

    def forward(self, hm_outputs, dm_outputs, targets, target_weights, cameras, limb):
        limb_length_loss = 0.0
        multiview_consistency_loss = 0.0
        joint_mse_loss = 0.0
        kps_3d_hat_list, pred_weights = [], []
        data = zip(hm_outputs, dm_outputs, targets, target_weights, cameras)
        for heatmap, depthmap, target, target_weight, camera in data:
            kps_25d = self.get_kps_25d(heatmap, depthmap)
            kps_25d_hat = self.get_scale_normalized_pose(kps_25d)
            kps_3d_hat = self.reconstruct_3d_kps(kps_25d_hat, camera)
            joint_mse_loss += self.criterion1(heatmap, target, target_weight)
            kps_3d_hat_list.append(kps_3d_hat)
        limb_length_loss += self.criterion2(kps_3d_hat_list)
        multiview_consistency_loss += self.criterion3(kps_3d_hat_list, target_weights)
        return joint_mse_loss + self.alpha * multiview_consistency_loss + self.beta * limb_length_loss


class JointMPJPELoss(nn.Module):
    def __init__(self):
        super(JointMPJPELoss, self).__init__()

    def forward(self, joint_3d, gt, joints_vis_3d=None, output_batch_mpjpe=False):
        """
        :param joint_3d: (batch, njoint, 3)
        :param gt: (batch, njoint, 3)
        :param joints_vis_3d: (batch, njoint, 1), values are 0,1
        :param output_batch_mpjpe: bool
        :return:
        """
        if joints_vis_3d is None:
            joints_vis_3d = torch.ones_like(joint_3d)[:,:,0:1]
        l2_distance = torch.sqrt(((joint_3d - gt)**2).sum(dim=2))
        joints_vis_3d = joints_vis_3d.view(*l2_distance.shape)
        masked_l2_distance = l2_distance * joints_vis_3d
        n_valid_joints = torch.sum(joints_vis_3d, dim=1)
        # if (n_valid_joints < 1).sum() > 0:
        n_valid_joints[n_valid_joints < 1] = 1  # avoid div 0
        avg_mpjpe = torch.sum(masked_l2_distance) / n_valid_joints.sum()
        if output_batch_mpjpe:
            return avg_mpjpe, masked_l2_distance, n_valid_joints.sum()
        else:
            return avg_mpjpe, n_valid_joints.sum()


class Joint2dSmoothLoss(nn.Module):
    def __init__(self):
        super(Joint2dSmoothLoss, self).__init__()
        factor = torch.as_tensor(8.0)
        alpha = torch.as_tensor(-10.0)
        self.register_buffer('factor', factor)
        self.register_buffer('alpha', alpha)

    def forward(self, joint_2d, gt, target_weight=None):
        """
        :param joint_2d: (batch*nview, njoint, 2)
        :param gt:
        :param target_weight: (batch*nview, njoint, 1)
        :return:
        """
        x = torch.sum(torch.abs(joint_2d - gt), dim=2)  # (batch*nview, njoint)
        x_scaled = ((x / self.factor) ** 2 / torch.abs(self.alpha-2) + 1) ** (self.alpha * 0.5) -1
        x_final = (torch.abs(self.alpha) - 2) / self.alpha * x_scaled

        loss = x_final
        if target_weight is not None:
            cond = torch.squeeze(target_weight) < 0.5
            loss = torch.where(cond, torch.zeros_like(loss), loss)
        loss_mean = loss.mean()
        return loss_mean * 1000.0