# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

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


def get_3d_joints(batch_heatmap, batch_depthmap):
    for h, d in zip(batch_heatmap.shape, batch_depthmap.shape):
        assert h is d, 'Heatmap and Depthmap have different shapes.'

    max_vals, row_idx = batch_heatmap.max(2)
    max_vals, col_idx = max_vals.max(2)

    depth = torch.take(batch_depthmap, row_idx)
    depth = torch.take(depth, col_idx)

    row_idx = row_idx.take(col_idx).type(torch.float32)
    col_idx = col_idx.float()
    points_3d = torch.cat((row_idx.unsqueeze(2), col_idx.unsqueeze(2), depth.unsqueeze(2)), 2)
    max_vals[max_vals < 0] = 0.0
    return points_3d, max_vals


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

    def get_normal_limb_length(self, output):
        parent = torch.index_select(output, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))

        dist = ((parent-child) ** 2).sum(dim=2) ** 0.5
        norm_dist = dist / dist[dist > 0].mean()
        return norm_dist

    def get_limb_weight(self, output_weight):
        parent = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))
        return parent.mul(child)

    def forward(self, output, target, output_weight):
        length_weight = self.get_limb_weight(output_weight)
        length_pred = self.get_normal_limb_length(output)
        loss = ((length_pred - target) ** 2).mul(length_weight).sum()
        return loss


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self):
        super(MultiViewConsistencyLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.human36_edge = [(0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
                             (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        self.human36_root = {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 0,
            5: 4,
            6: 5,
            7: 0,
            9: 7,
            11: 9,
            12: 11,
            14: 9,
            15: 14,
            16: 15,
            17: 9,
            18: 17,
            19: 18,
        }

    def get_pitch_yaw(self, output):
        x = output[:, :, 0]
        y = output[:, :, 1]
        z = output[:, :, 2]

        yaw = torch.atan2(x, z)
        padj = (x ** 2 + z ** 2) ** 0.5
        pitch = torch.atan2(padj, y)
        return torch.cat((pitch.unsqueeze(2), yaw.unsqueeze(2)), 2)

    def get_global_angle(self, output):
        grand_parent = torch.index_select(output, 1, torch.cuda.LongTensor(
            [self.human36_root[item[0]] for item in self.human36_edge]))
        parent = torch.index_select(output, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))

        parent_vector = grand_parent - parent
        child_vector = parent - child

        parent_py = self.get_pitch_yaw(parent_vector)
        child_py = self.get_pitch_yaw(child_vector)
        return child_py - parent_py

    def forward(self, joints_3ds):
        root_joints_3d = joints_3ds[0]
        num_edge = root_joints_3d.size(1)
        root_angle = self.get_global_angle(root_joints_3d)
        loss = 0
        cnt = 0
        for joints_3d in joints_3ds[1:]:
            angle = self.get_global_angle(joints_3d)
            loss += self.criterion(root_angle, angle)
            cnt += 1
        return (loss / num_edge) / cnt


class WeaklySupervisedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion1 = JointsMSELoss(use_target_weight)
        self.criterion2 = LimbLengthLoss()
        self.criterion3 = MultiViewConsistencyLoss()

        self.use_target_weight = use_target_weight
        self.alpha = 1.0
        self.beta = 1.0

    def forward(self, hm_outputs, dm_outputs, targets, target_weights, limb):
        joints_mse_loss = 0.0
        limb_length_loss = 0.0
        multiview_consistency_loss = 0.0
        joints_3ds, pred_weights = [], []
        for heatmap, depthmap, target, target_weight in zip(hm_outputs, dm_outputs, targets, target_weights):
            joints_3d, pred_weight = get_3d_joints(heatmap, depthmap)
            joints_mse_loss += self.criterion1(heatmap, target, target_weight)
            limb_length_loss += self.criterion2(joints_3d, limb, pred_weight)
            joints_3ds.append(joints_3d)
        #multiview_consistency_loss += self.criterion3(joints_3ds)
        return joints_mse_loss + limb_length_loss #+ multiview_consistency_loss


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