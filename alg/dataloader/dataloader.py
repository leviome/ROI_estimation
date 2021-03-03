import random

import numpy as np
import torch
from torch.utils import data

from alg.dataloader.dataset_interface import FOCUSDataSets
from alg.utils.sampler import TrainingSampler, InferenceSampler


class MutilScaleBatchCollator(object):
    def __init__(self, img_size, train):
        self.img_size = [a for a in range(min(img_size), max(img_size) + 32, 32)]
        self.train = train

    def normlize(self, img):

        img = np.float32(img) if img.dtype != np.float32 else img.copy()

        return img / 255.

    def process_image(self, meta, sized):
        images = []

        for info in meta:
            img = info['img']
            padding_img = img.copy()
            padding_img = self.normlize(padding_img)
            padding_img = torch.from_numpy(padding_img).permute(2, 0, 1).float()
            images.append(padding_img)

        return images

    def __call__(self, batch):
        meta = list(batch)
        if self.train:
            sized = random.choice(self.img_size)
        else:
            sized = sum(self.img_size) / float(len(self.img_size))

        images = self.process_image(meta, sized)
        batch_imgs = torch.cat([a.unsqueeze(0) for a in images])

        return batch_imgs, meta


def make_data_loader(list_path, img_size, train=False,
                     batch_size=4, num_workers=4):
    txt_folder = list_path
    data_set = FOCUSDataSets(txt_folder, train)
    collator = MutilScaleBatchCollator(img_size, train)
    if train:
        sampler = TrainingSampler(len(data_set), shuffle=train)
    else:
        sampler = InferenceSampler(len(data_set))

    data_loader = data.DataLoader(dataset=data_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collator,
                                  sampler=sampler,
                                  pin_memory=True
                                  )

    return data_loader
