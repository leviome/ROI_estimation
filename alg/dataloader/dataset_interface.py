import os.path as osp

import cv2
from torch.utils import data


class FOCUSDataSets(data.Dataset):
    def __init__(self, list_file, train=False):
        self.train = train
        self.data_folder = '../data/'
        self.label_str = []
        self.image_path = []

        with open(list_file) as f:
            lines = f.readlines()
        self.num_samples = len(lines)
        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])
            self.label_str.append(splited[1])

    def _get_label(self, gt_str, img_size):
        x, y = [float(i) for i in gt_str.split(',')]
        return [x * img_size[0], y * img_size[1]]

    def get_data(self, idx):
        file_name = osp.join(self.data_folder, self.image_path[idx])
        gt_str = osp.join(self.label_str[idx])
        img = cv2.imread(file_name)
        gt_list = self._get_label(gt_str, (img.shape[1], img.shape[0]))
        meta = dict()
        meta['img_width'] = img.shape[1]
        meta['img_height'] = img.shape[0]
        meta['boxlist'] = gt_list.copy()
        meta['img'] = img

        return meta

    def __getitem__(self, idx):
        meta = self.get_data(idx)
        return meta

    def __len__(self):
        return self.num_samples
