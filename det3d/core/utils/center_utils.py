# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou and Tianwei Yin 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from .circle_nms_jit import circle_nms

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep 


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def reorganize_test_cfg_for_multi_tasks(test_cfg, task_num_classes):
    """
    - for single float, copy for each task head;
    - for list, re-organize for multiple task heads.

    :param test_cfg:
      nms=dict(
        use_rotate_nms=False,
        # nms_pre_max_size=4096,
        # nms_post_max_size=500,
        # nms_iou_threshold=0.7,
        use_multi_class_nms=True,
        nms_pre_max_size=[2048, 1024, 1024],
        nms_post_max_size=[300, 150, 150],
        nms_iou_threshold=[0.8, 0.55, 0.55],
    ),
    rectifier=[0, 0, 0],
    score_threshold=0.1,
    :param task_num_classes: [int] * num_classes
    :return:
    """
    def reorganize_param(param):
        if isinstance(param, float) or isinstance(param, int):
            return [param] * len(task_num_classes)

        assert isinstance(param, list) or isinstance(param, tuple)
        assert len(param) == sum(task_num_classes)

        ret_list = [[]] * len(task_num_classes)
        flag = 0
        for k, num in enumerate(task_num_classes):
            ret_list[k] = list(param[flag:flag+num])
            flag += num
        return ret_list

    if test_cfg.get('rectifier', False) is not None:
        test_cfg['rectifier'] = reorganize_param(test_cfg['rectifier'])

    test_cfg['nms']['nms_pre_max_size'] = reorganize_param(test_cfg['nms']['nms_pre_max_size'])
    test_cfg['nms']['nms_post_max_size'] = reorganize_param(test_cfg['nms']['nms_post_max_size'])
    test_cfg['nms']['nms_iou_threshold'] = reorganize_param(test_cfg['nms']['nms_iou_threshold'])

    return test_cfg



def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_iou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    ious = volume_inter / volume_union
    ious = torch.clamp(ious, min=0, max=1.0)
    return ious


def bbox3d_overlaps_giou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    closure = outer[:, 0] * outer[:, 1] * outer_h

    gious = volume_inter / volume_union - (closure - volume_union) / closure
    gious = torch.clamp(gious, min=-1.0, max=1.0)
    return gious


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious

def bboxes_overlaps_ciou_bev(pred_boxes, gt_boxes):
    """ Calculate the circle IoU in bev space formed by 3D bounding boxes"""
    assert pred_boxes.shape[0] == gt_boxes.shape[0]
    assert pred_boxes.shape[1] == gt_boxes.shape[1] == 5
    cious = torch.zeros([pred_boxes.shape[0], ], device=pred_boxes.device, dtype=torch.float32)
    if pred_boxes.shape[0] == 0:
        return cious

    d = torch.sum(torch.pow(gt_boxes[:, 0:2] - pred_boxes[:, 0:2], 2), dim=1)
    # qr2 = torch.sum(pred_boxes[:, 3:5].clone() ** 2, dim=1)
    # gr2 = torch.sum(torch.pow(gt_boxes[:, 3:5], 2), dim=1)
    if torch.any(torch.isnan(d)):
        print("asdfsaf")

    d2 = d * d
    gr = gt_boxes[:, 3]
    gr2 = gr * gr

    index_inside = torch.nonzero(d <= torch.abs(pred_boxes[:, 3] - gr))
    if len(index_inside) > 0:
        cious[index_inside[:, 0]] = torch.minimum(pred_boxes[:, 3][index_inside[:, 0]] ** 2, gr2[index_inside[:, 0]]) / \
                                    torch.maximum(pred_boxes[:, 3][index_inside[:, 0]] ** 2, gr2[index_inside[:, 0]])

    index_insec = torch.nonzero((d > torch.abs(pred_boxes[:, 3] - gr)) & (d < (pred_boxes[:, 3] + gr)))
    if len(index_insec) > 0:
        # qr2_t = qr[index_insec[:, 0]] * qr[index_insec[:, 0]]
        # gr2_t = gr[index_insec[:, 0]] * gr[index_insec[:, 0]]
        x_t = (pred_boxes[:, 3][index_insec[:, 0]]**2 - gr2[index_insec[:, 0]] + d2[index_insec[:, 0]]) / (2 * d[index_insec[:, 0]])
        z_t = x_t * x_t
        y_t = torch.sqrt(torch.clamp(pred_boxes[:, 3][index_insec[:, 0]]**2 - z_t, min=0))

        overlaps = ((pred_boxes[:, 3][index_insec[:, 0]]**2) * torch.arcsin(torch.clamp(y_t / gr[index_insec[:, 0]], min=-1, max=1.)) +
                    gr2[index_insec[:, 0]] * torch.arcsin(torch.clamp(y_t / gr[index_insec[:, 0]], min=-1, max=1.)) -
                    y_t * (x_t + torch.sqrt(z_t + gr2[index_insec[:, 0]] - pred_boxes[:, 3][index_insec[:, 0]]**2)))
        cious[index_insec[:, 0]] = overlaps / (np.pi * pred_boxes[:, 3][index_insec[:, 0]]**2 + np.pi * gr2[index_insec[:, 0]] - overlaps)

        if torch.any(torch.isnan(cious)):
            print("asdfsaf")
    cious = torch.clamp(cious, min=0, max=1.0)
    return cious

def bboxes_overlaps_ciou(pred_boxes, gt_boxes):
    """ Calculate the circle IoU in bev space formed by 3D bounding boxes"""
    assert pred_boxes.shape[0] == gt_boxes.shape[0]
    assert pred_boxes.shape[1] == gt_boxes.shape[1] == 7

    cious = torch.zeros([pred_boxes.shape[0], ], device=pred_boxes.device, dtype=torch.float32)
    if pred_boxes.shape[0] == 0:
        return cious

    d2 = torch.pow(pred_boxes[:, 0:2] - gt_boxes[:, 0:2], 2).sum(-1)
    qr2 = torch.pow(pred_boxes[:, 3:5], 2).sum(-1)
    gr2 = torch.pow(gt_boxes[:, 3:5], 2).sum(-1)

    d = torch.sqrt(d2)
    qr = torch.sqrt(qr2)
    gr = torch.sqrt(gr2)

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inside_mask = d <= torch.abs(qr - gr)
    index_inside = torch.nonzero(inside_mask)
    if len(index_inside) > 0:
        overlaps_insides = torch.minimum(qr2[inside_mask], gr2[inside_mask]) * inter_h[inside_mask]  # * np.pi
        cious[index_inside[:, 0]] = overlaps_insides / (qr2[inside_mask] * gt_boxes[:, 5][inside_mask] +
                                                        gr2[inside_mask] * gt_boxes[:, 5][inside_mask] -
                                                        overlaps_insides + 1e-4)

    insec_mask = (d > torch.abs(qr - gr)) & (d < (qr + gr))
    index_insec = torch.nonzero(insec_mask)
    if len(index_insec) > 0:
        qr_t = qr[insec_mask]
        gr_t = gr[insec_mask]
        d_t = d[insec_mask]

        qr2_t = qr_t * qr_t
        gr2_t = gr_t * gr_t
        x_t = (qr2_t - gr2_t + d_t * d_t) / (2 * d_t)
        z_t = x_t * x_t
        y_t = torch.sqrt(torch.clamp(qr2_t - z_t, min=0.))

        overlaps = (qr2_t * torch.arcsin(torch.clamp(y_t / qr_t, max=1.)) +
                    gr2_t * torch.arcsin(torch.clamp(y_t / gr_t, max=1.)) -
                    y_t * (x_t + torch.sqrt(torch.clamp(z_t + gr2_t - qr2_t, min=0.)))) * inter_h[insec_mask]
        cious[index_insec[:, 0]] = overlaps / (np.pi * qr2_t * pred_boxes[:, 5][insec_mask] +
                                               np.pi * gr2_t * gt_boxes[:, 5][insec_mask] -
                                               overlaps + 1e-4)
    cious = torch.clamp(cious, min=0, max=1.0)
    return cious

def bboxes_overlaps_cdiou(pred_boxes, gt_boxes):
    """ Calculate the circle IoU in bev space formed by 3D bounding boxes"""
    assert pred_boxes.shape[1] == gt_boxes.shape[1] == 7
    # cious = torch.zeros([pred_boxes.shape[0], ], device=pred_boxes.device, dtype=torch.float32)

    d2 = torch.pow(pred_boxes[:, 0:2] - gt_boxes[:, 0:2], 2).sum(-1)
    qr2 = torch.pow(pred_boxes[:, 3:5], 2).sum(-1)
    gr2 = torch.pow(gt_boxes[:, 3:5], 2).sum(-1)

    d = torch.sqrt(d2)
    qr = torch.sqrt(qr2)
    gr = torch.sqrt(gr2)

    cdious = -1. * d2 / torch.pow(d + qr + gr, 2)

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inside_mask = d <= torch.abs(qr - gr)
    index_inside = torch.nonzero(inside_mask)

    d2 = d2[inside_mask]
    qr2 = qr2[inside_mask]
    gr2 = gr2[inside_mask]
    overlaps_insides = torch.minimum(qr2, gr2) * inter_h[inside_mask]  # * np.pi
    cdious[index_inside[:, 0]] = overlaps_insides / (
            qr2 * gt_boxes[:, 5][inside_mask] + gr2 * gt_boxes[:, 5][inside_mask] - overlaps_insides) - \
            d2 / torch.maximum(qr2, gr2)

    insec_mask = (~inside_mask) & (d < (qr + gr))
    index_insec = torch.nonzero(insec_mask)
    qr = qr[insec_mask]
    gr = gr[insec_mask]
    d = d[insec_mask]

    d2 = d * d
    qr2 = qr * qr
    gr2 = gr * gr
    x = (qr2 - gr2 + d2) / (2 * d)
    z = x * x
    y = torch.sqrt(qr2 - z)

    overlaps = (qr2 * torch.arcsin(y / qr) + gr2 * torch.arcsin(y / gr) -
                y * (x + torch.sqrt(z + gr2 - qr2))) * inter_h[insec_mask]
    cdious[index_insec[:, 0]] = overlaps / (
            np.pi * qr2 * pred_boxes[:, 5][insec_mask] + np.pi * gr2 * gt_boxes[:, 5][insec_mask] - overlaps) - \
            d2 / torch.pow(d + qr + gr, 2)
    cdious = torch.clamp(cdious, min=-1, max=1.)
    return cdious