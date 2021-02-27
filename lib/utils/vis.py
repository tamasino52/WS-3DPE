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
import torchvision
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.inference import get_max_preds
from lib.core.inference import PoseReconstructor
from lib.utils.procrustes import compute_similarity_transform_torch as procrustes_trans


class PoseVisualizer(PoseReconstructor):
    def __init__(self):
        super(PoseVisualizer, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.h36m_pair = [
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6)]
        self.union_pair = [
            (0, 7), (7, 9), (9, 11), (11, 12), (9, 14), (14, 15), (15, 16), (9, 17), (17, 18),
            (18, 19), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
        self.cmap = ['r', 'g', 'b', 'y']

    def softmax2d(self, x):
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H))
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def normalize_img(self, image):
        min = float(image.min())
        max = float(image.max())

        image.add_(-min).div_(max - min + 1e-5)
        image = image.mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def draw_3d_plot(self, ax, kps_3d, color='r'):
        i1 = [self.union_pair[idx][0] for idx in range(len(self.union_pair))]
        i2 = [self.union_pair[idx][1] for idx in range(len(self.union_pair))]

        for idx in range(len(self.union_pair)):
            x = np.array([kps_3d[i1[idx], 0], kps_3d[i2[idx], 0]])
            y = np.array([kps_3d[i1[idx], 2], kps_3d[i2[idx], 2]])
            z = -np.array([kps_3d[i1[idx], 1], kps_3d[i2[idx], 1]])
            ax.plot(x, y, z, linewidth=2, color='k')

        ax.scatter(kps_3d[i1, 0], kps_3d[i1, 2], -kps_3d[i1, 1], marker='o', color=color)
        ax.scatter(kps_3d[i2, 0], kps_3d[i2, 2], -kps_3d[i2, 1], marker='o', color=color)
        '''
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        '''

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        return ax

    def save_batch_kps_3d(self, batch_image_list, batch_heatmap_list, batch_depthmap_list, camera_list, file_name):
        '''
        batch_image: [batch_size, channel, height, width]
        batch_heatmap_list: [['batch_size, num_joints, height, width] ...]
        batch_depthmap_list: [['batch_size, num_joints, height, width] ...]

        file_name: saved file name
        '''

        view_size = len(batch_heatmap_list)
        batch_size = batch_heatmap_list[0].size(0)

        fig = plt.figure(figsize=(12 * view_size, 6 * batch_size))

        ax_list = []
        for batch in range(batch_size):
            ax = fig.add_subplot(batch_size, 2 * view_size + 1, ((batch + 1) * (2 * view_size + 1)), projection='3d')
            ax.set_title('Rigid combination')
            ax_list.append(ax)

        data = zip(batch_image_list, batch_heatmap_list, batch_depthmap_list, camera_list)
        for view, (img, hm, dm, cam) in enumerate(data):
            kps_3d = self.infer(hm, dm, cam)
            if view is 0:
                root_kps_3d = kps_3d

            for batch in range(batch_size):
                image = img[batch]

                ax = fig.add_subplot(batch_size, 2 * view_size + 1, (batch * (2 * view_size + 1)) + 2 * view + 1)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

                image = self.normalize_img(image)
                ax.imshow(image, vmin=0, vmax=255)
                ax = fig.add_subplot(batch_size, 2 * view_size + 1, (batch * (2 * view_size + 1)) + 2 * view + 2, projection='3d')
                self.draw_3d_plot(ax, kps_3d[batch].detach().cpu().numpy(), self.cmap[view])
                kps_trans = procrustes_trans(kps_3d[batch], root_kps_3d[batch])
                self.draw_3d_plot(ax_list[batch], kps_trans.detach().cpu().numpy(), self.cmap[view])

        plt.savefig(file_name)
        plt.close(fig)


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_depthmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output, depth=None, prefix=None):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints_2d_transformed'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED and depth is not None:
        save_batch_depthmaps(
            input, depth, '{}_dm_pred.jpg'.format(prefix)
        )


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d_vis[n, i1, 0] > 0 and kpt_3d_vis[n, i2, 0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n, i1, 0] > 0:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d_vis[n, i2, 0] > 0:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def show3Dpose(channels, ax, radius=40, mpii=2, lcolor='#ff0000', rcolor='#0000ff'):
    vals = channels

    if mpii == 0: # h36m with mpii joints
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif mpii == 1: # only mpii
        connections = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                       [7, 8], [8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    else: # default h36m
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)

    RADIUS = radius  # space around the subject
    if mpii == 1:
        xroot, yroot, zroot = vals[6, 0], vals[6, 1], vals[6, 2]
    else:
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")