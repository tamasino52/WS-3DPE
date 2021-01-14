# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import numpy as np

import torch
import torch.optim as optim


def ortho2rotation(poses):
  r"""
  poses: batch x 6
  """
  def normalize_vector(v):
    r"""
    Batch x 3
    """
    v_mag = torch.sqrt((v**2).sum(1, keepdim=True))
    v_mag = torch.clamp(v_mag, min=1e-8)
    v = v / v_mag
    return v

  def cross_product(u, v):
    r"""
    u: batch x 3
    v: batch x 3
    """
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    i = i[:, None]
    j = j[:, None]
    k = k[:, None]
    return torch.cat((i, j, k), 1)

  def proj_u2a(u, a):
    r"""
    u: batch x 3
    a: batch x 3
    """
    inner_prod = (u * a).sum(1, keepdim=True)
    norm2 = (u**2).sum(1, keepdim=True)
    norm2 = torch.clamp(norm2, min=1e-8)
    factor = inner_prod / norm2
    return factor * u

  x_raw = poses[:, 0:3]
  y_raw = poses[:, 3:6]

  x = normalize_vector(x_raw)
  y = normalize_vector(y_raw - proj_u2a(x, y_raw))
  z = cross_product(x, y)

  x = x[:, :, None]
  y = y[:, :, None]
  z = z[:, :, None]
  return torch.cat((x, y, z), 2)


def argmin_se3_squared_dist(X, Y):
  """
  X: torch tensor N x 3
  Y: torch tensor N x 3
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  mux = X.mean(0, keepdim=True)
  muy = Y.mean(0, keepdim=True)

  Sxy = (Y - muy).t().mm(X - mux) / len(X)
  U, D, V = Sxy.svd()
  # svd = gesvd.GESVDFunction()
  # U, S, V = svd.apply(Sxy)
  # S[-1, -1] = U.det() * V.det()
  S = torch.eye(3)
  if U.det() * V.det() < 0:
    S[-1, -1] = -1

  R = U.mm(S.mm(V.t()))
  t = muy.squeeze() - R.mm(mux.t()).squeeze()
  return R, t


def weighted_procrustes(X, Y, w, eps):
  """
  X: torch tensor N x 3
  Y: torch tensor N x 3
  w: torch tensor N
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  W1 = torch.abs(w).sum()
  w_norm = w / (W1 + eps)
  mux = (w_norm * X).sum(0, keepdim=True)
  muy = (w_norm * Y).sum(0, keepdim=True)

  # Use CPU for small arrays
  Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
  U, D, V = Sxy.svd()
  S = torch.eye(3).double()
  if U.det() * V.det() < 0:
    S[-1, -1] = -1

  R = U.mm(S.mm(V.t())).float()
  t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
  return R, t


class Transformation(torch.nn.Module):
  def __init__(self, R_init=None, t_init=None):
    torch.nn.Module.__init__(self)
    rot_init = torch.rand(1, 6)
    trans_init = torch.zeros(1, 3)
    if R_init is not None:
      rot_init[0, :3] = R_init[:, 0]
      rot_init[0, 3:] = R_init[:, 1]
    if t_init is not None:
      trans_init[0] = t_init

    self.rot6d = torch.nn.Parameter(rot_init)
    self.trans = torch.nn.Parameter(trans_init)

  def forward(self, points):
    rot_mat = ortho2rotation(self.rot6d)
    return points @ rot_mat[0].t() + self.trans