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

from multiviews.procrustes import batch_compute_similarity_transform_torch


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

    batch_size = batch_heatmap.shape[0]
    num_joints = batch_heatmap.shape[1]
    width = batch_heatmap.shape[3]
    heatmap_reshaped = batch_heatmap.view(batch_size, num_joints, -1)
    depthmap_reshaped = batch_depthmap.view(batch_size, num_joints, -1)

    maxvals, idx = torch.max(heatmap_reshaped, 2)
    depths = torch.take(depthmap_reshaped, idx)

    maxvals = maxvals.view(batch_size, num_joints, 1)
    idx = idx.view(batch_size, num_joints, 1)
    depths = depths.view(batch_size, num_joints)

    preds = idx.repeat_interleave(3, 2).type(torch.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)
    preds[:, :, 2] = depths

    pred_mask = torch.gt(maxvals, 0.0).repeat_interleave(3, 2)
    pred_mask = pred_mask.type(torch.float32)

    preds *= pred_mask
    return preds, maxvals


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
        norm_dist = dist / dist.mean()
        return norm_dist

    def get_limb_weight(self, output_weight):
        parent = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[0] for item in self.human36_edge]))
        child = torch.index_select(output_weight, 1, torch.cuda.LongTensor([item[1] for item in self.human36_edge]))
        return parent.mul(child)

    def forward(self, output, target, output_weight):
        length_weight = self.get_limb_weight(output_weight)
        length_pred = self.get_normal_limb_length(output)
        batch_size = length_pred.shape[0]
        num_edges = length_pred.shape[1]
        length_pred = length_pred.reshape((batch_size, num_edges, -1))
        length_gt = target.reshape((batch_size, num_edges, -1))
        loss = ((length_pred - length_gt) ** 2).mul(length_weight).sum() / batch_size
        return loss


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(MultiViewConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, joints_3ds, output_weights, target_weights):
        root_joints_3d = joints_3ds[0]
        root_output_weight = output_weights[0]
        root_target_weight = target_weights[0]
        for joints_3d, output_weight, target_weight in zip(joints_3ds[1:], output_weights[1:], target_weights[1:]):
            pass


class WeaklySupervisedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion1 = JointsMSELoss(use_target_weight)
        self.criterion2 = LimbLengthLoss()
        self.criterion3 = MultiViewConsistencyLoss(use_target_weight)

        self.use_target_weight = use_target_weight
        self.alpha = 10.0
        self.beta = 100.0

    def forward(self, hm_outputs, dm_outputs, targets, target_weights, limb):
        joints_mse_loss = 0
        limb_length_loss = 0
        multiview_consistency_loss = 0
        cnt = 0
        joints_3ds, pred_weights = [], []
        for heatmap, depthmap, target, target_weight in zip(hm_outputs, dm_outputs, targets, target_weights):
            joints_3d, pred_weight = get_3d_joints(heatmap, depthmap)
            joints_mse_loss += self.criterion1(heatmap, target, target_weight)
            limb_length_loss += self.criterion2(joints_3d, limb, pred_weight)
            joints_3ds.append(joints_3d)
            pred_weights.append(pred_weight)
            cnt += 1
        multiview_consistency_loss += self.criterion3(joints_3ds, pred_weights, target_weights)
        return joints_mse_loss / cnt + self.alpha * limb_length_loss / cnt + self.beta * multiview_consistency_loss
