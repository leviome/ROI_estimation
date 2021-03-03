#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# @Author: liwei
# Copyright (c) 2020 Intel
# Created by Liwei Liao liwei.liao@intel.com
# --------------------------------------------------------
import cv2
import numpy as np


def gaussian_blur_array(hm, blur_range=(3, 3)):
    blurred_hm = cv2.GaussianBlur(hm, blur_range, 6)
    a = max(np.amax(blurred_hm), 1e-6)
    blurred_hm /= a
    return blurred_hm


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


if __name__ == '__main__':
    po = [1677, 2359]
    poi = [po[0] / 5120, po[1] / 3072]
    one_hot_map = p2array([30 / 128, 0.128])
    blurred = gaussian_blur_array(one_hot_map)
    print(blurred)
