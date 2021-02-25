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
from lib.utils.procrustes import batch_compute_similarity_transform_torch as procrustes_align
from lib.core.inference import PoseReconstructor


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


class JointMPJPELoss(nn.Module):
    def __init__(self):
        super(JointMPJPELoss, self).__init__()

    def forward(self, joint_3d, gt, joints_vis_3d=None, output_batch_mpjpe=False):
        """
        :param joint_3d: (batch, njoint, 3)
        :param gt:
        :param joints_vis_3d: (batch, njoint, 1), values are 0,1
        :param output_batch_mpjpe: bool
        :return:
        """
        if joints_vis_3d is None:
            joints_vis_3d = torch.ones_like(joint_3d)[:, :, 0:1]
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
            return avg_mpjpe#, n_valid_joints.sum()


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

    def get_limb_length(self, kps_3d):
        human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                        (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        parent = kps_3d[:, [item[0] for item in human36_edge], :]
        child = kps_3d[:, [item[1] for item in human36_edge], :]
        dist = (((parent - child) ** 2).sum(2) ** 0.5)
        return dist

    '''
        def get_limb_weight(self, output_weight):
        parent = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))
        return parent.mul(child)
    '''

    def forward(self, joints_3d, avg_limb):
        loss = 0.0
        length_gt = avg_limb.clone()
        length_pred = self.get_limb_length(joints_3d)
        loss += self.criterion(length_gt, length_pred)
        return loss


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self):
        super(MultiViewConsistencyLoss, self).__init__()
        self.criterion = JointMPJPELoss()
        self.human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                             (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        self.vis = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19]

    def forward(self, joints_3ds, target_weights):
        loss = 0.0
        for batch_root_joints_3d, batch_root_target_weight in zip(joints_3ds, target_weights):
            for batch_joints_3d, batch_target_weight in zip(joints_3ds, target_weights):
                '''
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
                '''
                #trans_kps_3d = criterion_procrustes(batch_joints_3d, batch_root_joints_3d)
                for joints_3d, root_joints_3d in zip(batch_joints_3d, batch_root_joints_3d):
                    loss += criterion_procrustes(joints_3d[self.vis, :], root_joints_3d[self.vis, :]) #self.criterion(trans_kps_3d, batch_root_joints_3d, batch_target_weight)

        return loss


class WeaklySupervisedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion1 = JointsMSELoss(use_target_weight)
        self.criterion2 = LimbLengthLoss()
        self.criterion3 = MultiViewConsistencyLoss()
        self.mse = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.alpha = 0.01
        self.beta = 0.1
        self.pose_reconstructor = PoseReconstructor()

    def forward(self, hm_outputs, dm_outputs, targets, target_weights, cameras, limb):
        limb_length_loss = 0.0
        multiview_consistency_loss = 0.0
        joint_mse_loss = 0.0

        kps_3d_hat_list = []

        data = zip(hm_outputs, dm_outputs, targets, target_weights, cameras)
        for heatmap, depthmap, target, target_weight, camera in data:
            joint_mse_loss += self.criterion1(heatmap, target, target_weight)
            kps_3d_hat = self.pose_reconstructor.infer(heatmap, depthmap, camera)
            limb_length_loss += self.criterion2(kps_3d_hat, limb)
            kps_3d_hat_list.append(kps_3d_hat)
        multiview_consistency_loss += self.criterion3(kps_3d_hat_list, target_weights)
        total_loss = joint_mse_loss + self.beta * limb_length_loss + self.alpha * multiview_consistency_loss
        return total_loss, joint_mse_loss, multiview_consistency_loss, limb_length_loss


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