import os
import os.path as osp
import time

import torch
from torch.cuda import amp

from alg.dataloader.dataloader import make_data_loader
from alg.nn.rep_block import RepFocusModel


def train_step(epochs, model, train_loader, test_loader, optim, device='cuda'):
    x_placeholder, _ = next(iter(train_loader))
    scaler = amp.GradScaler(enabled=True)
    for epoch in range(epochs):
        print('epoch =', epoch)
        for idx, (img, gt_info) in enumerate(train_loader):
            optim.zero_grad()
            img = img.to(device)
            loss = model(img, gt_info)
            print(int(loss * 10), end=' ')
            scaler.scale(loss).backward()
            optim.step()
        # val_step(model, test_loader, device=device)
        calender = time.localtime(time.time())
        month = calender.tm_mon
        day = calender.tm_mday
        hour = calender.tm_hour
        minute = calender.tm_min
        folder_name = 'ckpt/model_' + str(month) + '_' + str(day)
        if not osp.exists(folder_name):
            os.system('mkdir ' + folder_name)
        if epoch % 10 == 0:
            model_name = 'ckpt_' + str(hour) + '_' + str(minute) + '_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), osp.join(folder_name, model_name))


def val_step(model, test_loader, device='cuda'):
    model.eval()
    all_not = 0
    for idx, (img, gt_info) in enumerate(test_loader):
        img = img.to(device)
        not_same = model.estimate(img, gt_info)
        all_not += not_same
    print(all_not)
    model.train()


def train(load_ckpt=False):
    epochs = 300
    lr = [0.5e-3, 1e-5]
    bs = 8
    device = 'cuda'
    train_root = '../data/'
    img_size = [1280, 768]

    model = RepFocusModel(num_blocks=[2, 4, 14, 1],
                          width_multiplier=[0.75, 0.75, 0.75, 2],
                          override_groups_map=None,
                          deploy=False)

    if load_ckpt:
        checkpoint = torch.load('ckpt/model.pth')
        data_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(data_dict, strict=True)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr[0],
                                 weight_decay=5e-5)

    train_loader = make_data_loader(osp.join(train_root, 'train.txt'),
                                    img_size=img_size,
                                    batch_size=bs,
                                    train=True,
                                    )
    test_loader = make_data_loader(osp.join(train_root, 'test.txt'),
                                   img_size=img_size,
                                   batch_size=bs,
                                   train=False,
                                   )

    train_step(epochs, model, train_loader, test_loader, optimizer, device=device)


if __name__ == '__main__':
    train(load_ckpt=False)
