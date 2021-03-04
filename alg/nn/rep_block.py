import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


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


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
        You can get the equivalent kernel and bias at any time and do whatever you want,
        for example, apply some penalties or constraints during training, just like you do to the other models.
        May be useful for quantization or pruning.
        :return:
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()


class RepFocusModel(nn.Module):
    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepFocusModel, self).__init__()
        self.img_size = (1280, 768)

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(1, num_blocks[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x, target=None):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.max_pool2d(out, kernel_size=(3, 5), stride=(3, 5))
        if target is not None:
            points = [gt['boxlist'] for gt in target]
            gt_map = self.gaussian_blur_gt(points, out)
            loss = F.mse_loss(out, gt_map, reduction='sum')
            return loss
        else:
            feature_map = F.avg_pool2d(out, kernel_size=2, stride=1)
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


if __name__ == '__main__':
    model = RepFocusModel(num_blocks=[2, 4, 14, 1],
                          width_multiplier=[0.75, 0.75, 0.75, 2], override_groups_map=None,
                          deploy=True)
    print(model)
    image_black = np.array(
        np.random.randint(0, 256, size=(768, 1280, 3), dtype='uint8'))
    processed_frame = torch.from_numpy(image_black)
    processed_frame = processed_frame.float().div(255.0)
    processed_frame = processed_frame.unsqueeze(0)
    processed_frame = processed_frame.permute(0, 3, 1, 2)
    a = model(processed_frame)
    print(a.shape)
