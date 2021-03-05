import os.path as osp

import numpy as np
import torch
from tqdm import tqdm

from alg.dataloader.dataloader import make_data_loader
from alg.nn.focus_model import ROIEstimator


def get_crop_tile_from_heat_map(hm, img_size=(1280, 768), crop_ratio=8):
    index = hm.argmax()
    c, _, length = hm.shape
    row = index // length
    col = index - length * row
    hm[0][row][col] = 0
    x_coors = np.linspace(0, img_size[0], num=length + 2)
    y_coors = np.linspace(0, img_size[1], num=length + 2)
    left = x_coors[col]
    top = y_coors[row]
    right = left + img_size[0] / crop_ratio
    bottom = top + img_size[1] / crop_ratio

    if right > img_size[0]:
        left = int((1 - 1 / crop_ratio) * img_size[0])
        right = img_size[0]
    if bottom > img_size[1]:
        top = int((1 - 1 / crop_ratio) * img_size[1])
        bottom = img_size[1]

    return [top, left, bottom, right]


def run_eval():
    model = ROIEstimator()
    print('loading model...')
    checkpoint = torch.load('16x16.pth')
    data_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(data_dict, strict=True)
    model.cuda()
    model.eval()

    print('loading test dataset...')
    train_root = '../data/'
    test_loader = make_data_loader(osp.join(train_root, 'test.txt'),
                                   img_size=[1280, 768],
                                   batch_size=4,
                                   train=False)
    all_elements = 0
    hit_elements = 0
    for idx, (img, gt_info) in tqdm(enumerate(test_loader)):
        img = img.to('cuda')
        res = model(img)
        points = [gt['boxlist'] for gt in gt_info]
        for i, point in enumerate(points):
            all_elements += 1
            top, left, bottom, right = get_crop_tile_from_heat_map(res[i])
            if left < point[0] < right and top < point[1] < bottom:
                hit_elements += 1
    print(hit_elements / all_elements)


if __name__ == '__main__':
    run_eval()
