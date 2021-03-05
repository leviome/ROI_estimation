# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from alg.nn.focus_backbone import NNBackbone


def p2array(p, size=(128, 128)):
    """

    :param p: normalized (0-1)
    :param size:
    :return:
    """
    x, y = p
    w, h = size
    x_ = int(x * w)
    y_ = int(y * h)
    arr = np.zeros(size, dtype=np.float32)
    arr[y_][x_] = 1
    return arr


def gaussian_blur_array(hm, blur_range=(3, 3)):
    blurred_hm = cv2.GaussianBlur(hm, blur_range, 6)
    a = max(np.amax(blurred_hm), 1e-6)
    blurred_hm /= a
    return blurred_hm


def get_crop_tile_from_index(index, img_size=(5120, 3072)):
    """
    8x8
    :param hm:
    :param img_size:
    :return:
    """
    row = index // 7
    col = index - 7 * row
    x_coors = np.linspace(0, img_size[0], num=9)
    y_coors = np.linspace(0, img_size[1], num=9)
    left = x_coors[col]
    top = y_coors[row]
    right = left + img_size[0] / 4
    bottom = top + img_size[1] / 4
    return [top, left, bottom, right]


def get_crop_index(center, image_size=(5120, 3072), crop_size=(1866, 1120)):
    x, y = center
    if x == -1 or y == -1:
        return [-1000, -1000, -1000, -1000]
    img_w, img_h = image_size
    crop_w, crop_h = crop_size
    top = int(y - crop_h / 2)
    end = int(y + crop_h / 2)
    left = int(x - crop_w / 2)
    right = int(x + crop_w / 2)
    if int(y - crop_h / 2) < 0:
        top = 0
        end = crop_h
    if int(x - crop_w / 2) < 0:
        left = 0
        right = crop_w
    if int(y + crop_h / 2) > img_h:
        top = img_h - crop_h
        end = img_h
    if int(x + crop_w / 2) > img_w:
        left = img_w - crop_w
        right = img_w
    return [top, end, left, right]


class ROIEstimator(nn.Module):
    def __init__(self, input_size=(1280, 768)):
        super(ROIEstimator, self).__init__()
        self.img_size = input_size
        self.backbone = NNBackbone()

    def forward(self, x, target=None):
        B, c, h, w = x.shape
        self.img_size = (w, h)
        feature_map = self.backbone(x)
        if target is not None:
            points = [gt['boxlist'] for gt in target]
            gt_map = self.gaussian_blur_gt(points, feature_map)
            loss = F.mse_loss(feature_map, gt_map, reduction='sum')
            return loss
        else:
            # feature_map = F.avg_pool2d(feature_map, kernel_size=32, stride=32)
            feature_map = F.avg_pool2d(feature_map, kernel_size=3, stride=1)
            return feature_map

    def gaussian_blur_gt(self, points, fm, blur=True):
        res = np.zeros_like(fm.detach().cpu().numpy())
        w, h = self.img_size
        for i, p in enumerate(points):
            gt_slice = p2array([p[0] / w, p[1] / h], size=(8, 8))
            if blur:
                gt_slice = gaussian_blur_array(gt_slice, blur_range=(3, 3))
            gt_slice = gt_slice[None]  # [] -> [[]]
            res[i] = gt_slice
        gt_tensor = torch.from_numpy(res)
        return gt_tensor.cuda()

    def estimate(self, x, target=None, raw_image_size=(5120, 3072)):
        heat_map = self.forward(x)
        not_same = 0
        if target is not None:
            points = [gt['boxlist'] for gt in target]
            for i, point in enumerate(points):
                ind = heat_map[i].argmax()
                top, left, _, _ = get_crop_tile_from_index(ind, img_size=raw_image_size)
                xr = left + raw_image_size[0] / 8
                yr = top + raw_image_size[1] / 8
                top, end, left, right = get_crop_index([xr, yr], image_size=raw_image_size, crop_size=(1600, 960))
                if left < point[0] < right and top < point[1] < end:
                    pass
                else:
                    not_same += 1
            return not_same
        else:
            ind = heat_map.argmax()
            top, left, _, _ = get_crop_tile_from_index(ind, img_size=raw_image_size)
            xr = left + raw_image_size[0] / 8
            yr = top + raw_image_size[1] / 8
            top, end, left, right = get_crop_index([xr, yr], image_size=raw_image_size, crop_size=(1600, 960))
            area = [top, end, left, right]
            return area



