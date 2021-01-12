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


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
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


def get_3d_joints(heatmap, depthmap):
    for h, d in zip(heatmap.shape, depthmap.shape):
        assert h is d, 'Heatmap and Depthmap have different shapes.'

    batch_size = heatmap.shape[0]
    num_joints = heatmap.shape[1]
    width = heatmap.shape[3]
    heatmap_reshaped = heatmap.view(batch_size, num_joints, -1)
    depthmap_reshaped = depthmap.view(batch_size, num_joints, -1)

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
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

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
    def __init__(self, use_target_weight):
        super(LimbLengthLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.human36_skeleton = [
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6)]

    def euclidean_distance(self, pt1, pt2):
        distance = 0
        for i in range(len(pt1)):
            distance += (pt1[i] - pt2[i]) ** 2
        return distance ** 0.5

    def get_normal_limb_length(self, pose):
        batch_size = pose.shape[0]
        edges = torch.zeros([batch_size, len(self.human36_skeleton), 1], dtype=torch.float32)

        for i, edge in enumerate(self.human36_skeleton):
            parent, child = edge
            edges[:, i, 0] = self.euclidean_distance(pose[:, parent], pose[:, child])
        mean_length = sum(edges) / len(edges)
        edges = [edge / mean_length for edge in edges]
        return edges

    def forward(self, output, target, target_weight):
        output = self.get_normal_limb_length(output)
        batch_size = output.size(0)
        num_edges = output.size(1)
        joints_pred = output.reshape((batch_size, num_edges, -1)).split(1, 1)
        joints_gt = target.reshape((batch_size, num_edges, -1)).split(1, 1)
        loss = 0

        for idx in range(num_edges):
            joints_pred = joints_pred[idx].squeeze()
            joints_gt = joints_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    joints_pred.mul(target_weight[:, idx]),
                    joints_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(joints_pred, joints_gt)

        return loss / num_edges


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(MultiViewConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, joints_3ds, target_weights):
        batch_size = output.size(0)
        num_joints = output.size(1)
        pose_output = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        pose_target = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            pose_output = pose_output[idx].squeeze()
            pose_target = pose_target[idx].squeeze()
            pose_length = self.mse(pose_output, pose_target)

            loss += target_weight[num_joints] * self.mse(pose_length, normal_limbs)

        return loss / num_joints


class WeaklySupervisedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(WeaklySupervisedLoss, self).__init__()
        self.criterion1 = JointsMSELoss(use_target_weight)
        self.criterion2 = LimbLengthLoss(use_target_weight)
        self.criterion3 = MultiViewConsistencyLoss(use_target_weight)

        self.use_target_weight = use_target_weight
        self.alpha = 10.0
        self.beta = 100.0

    def forward(self, hm_outputs, dm_outputs, targets, target_weights, limb):
        joints_mse_loss = 0
        limb_length_loss = 0
        multiview_consistency_loss = 0
        cnt = 0
        joints_3ds, credibilities = [], []
        for heatmap, depthmap, target, target_weight in zip(hm_outputs, dm_outputs, targets, target_weights):
            joints_3d, credibility = get_3d_joints(heatmap, depthmap)
            joints_mse_loss += self.criterion1(heatmap, target, target_weight)
            limb_length_loss += self.criterion2(joints_3d, limb, target_weight.mul(credibilities))
            joints_3ds.append(joints_3d)
            credibilities.append(credibility)
            cnt += 1
        multiview_consistency_loss += self.criterion3(joints_3ds, target_weight.mul(credibilities))
        return joints_mse_loss / cnt + self.alpha * limb_length_loss / cnt + self.beta * multiview_consistency_loss
