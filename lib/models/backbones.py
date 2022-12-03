# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

from lib.models.modules import Bottleneck, ResNet_plus2

eps = 1e-5


class ResNet50(nn.Module):
    def __init__(self, used_layers=[2, 3, 4]):
        super(ResNet50, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3], used_layers=used_layers)

    def forward(self, x):
        x_stages, x = self.features(x)
        return x_stages, x


if __name__ == '__main__':
    z = torch.ones(32, 3, 127, 127)
    x = torch.ones(32, 3, 256, 256)

    model = ResNet50()
    zf = model(z)
    xf = model(x)

    if type(zf) is list:
        print([i.shape for i in zf])
        print([i.shape for i in xf])
    else:
        print(zf.shape)
        print(xf.shape)
