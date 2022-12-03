# -*- encoding: utf-8 -*-

"""
@Author : Soarkey
@Date   : 2021/10/26
@Desc   : cascade depth-wise cross correlation module
"""

import torch
import torch.nn as nn

from lib.core.config import config
from lib.models.blocked_co_attn import BCA
from lib.models.connect import DepthwiseXCorr, GroupDW, matrix


# cascade depth-wise cross correlation
class cascade_depthwise_corr(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(cascade_depthwise_corr, self).__init__()
        # coarse dw-xcorr
        self.first_dw_xcorr = DepthwiseXCorr(in_channels, out_channels, out_channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # fine dw-xcorr
        self.second_dw_xcorr = DepthwiseXCorr(in_channels, out_channels, out_channels)
        # fusion weights
        self.alpha = nn.Parameter(0.2 * torch.ones(1).cuda())
        self.beta = nn.Parameter(0.8 * torch.ones(1).cuda())

    def forward(self, kernel, search, corr=None):
        """
        Args:
            kernel: [32,256,7,7]
            search: [32,256,31,31]
            corr:   [32,256,25,25]
        Returns:
            kernel: [32,256,7,7]
            search: [32,256,31,31]
            phi_stage_2: [32,256,25,25]
        """
        if corr == None:
            # coarse dw-xcorr
            phi_stage_1 = self.first_dw_xcorr(kernel, search)
        else:
            phi_stage_1 = corr
        # upsample and fusion
        phi_up = self.upsample(phi_stage_1)
        search = self.fusion_layer(self.alpha * search + self.beta * phi_up)
        # fine dw-xcorr
        phi_stage_2 = phi_stage_1 + self.second_dw_xcorr(kernel, search)
        return kernel, search, phi_stage_2


# 级联互相关模块
class cascade_tower(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stage=1):
        super(cascade_tower, self).__init__()

        # attention module
        if config.TRAIN.ATTN.TYPE is not None:
            if config.TRAIN.ATTN.TYPE == 'BCA':
                self.attn = BCA(out_channels)
            else:
                raise NotImplementedError("config.TRAIN.ATTN")

        # cascade depth-wise correlation module
        for i in range(stage):
            self.add_module('cascade' + str(i + 1), cascade_depthwise_corr())
        self.stage = stage

        # reg corr
        self.reg_encode = matrix(in_channels, out_channels, type='reg')
        self.reg_dw = GroupDW()

        # cls/reg head
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
        )

        # adjust scale
        self.adjust = nn.Parameter(1.0 * torch.ones(1).cuda())
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, kernel):
        """
        Args:
            search: [32,256,31,31]
            kernel: [32,256,7,7]
        Returns:
                       x,          cls,          cls_f,            reg
               bbox_pred,     cls_pred,    cls_feature,    reg_feature
            [32,4,25,25], [32,1,25,25], [32,256,25,25], [32,256,25,25]
        """
        # attention module
        if config.TRAIN.ATTN.TYPE is not None:
            kernel, search = self.attn(kernel, search)

        # cascade module
        k, s, cls_f = kernel, search, None
        for i in range(self.stage):
            cascade_layer = getattr(self, 'cascade' + str(i + 1))
            k, s, cls_f = cascade_layer(k, s, cls_f)

        # cls head
        cls = self.cls_pred(cls_f)

        # reg
        reg_dw = self.bbox_tower(self.reg_dw(*self.reg_encode(kernel, search)))
        # adjust for loc sacle
        x = torch.exp(self.adjust * self.bbox_pred(reg_dw) + self.bias)

        return x, cls, cls_f, reg_dw


if __name__ == '__main__':
    zf = torch.randn([32, 256, 7, 7]).cuda()
    xf = torch.randn([32, 256, 31, 31]).cuda()

    model = cascade_tower(in_channels=256, out_channels=256).cuda()
    x, cls, cls_f, reg_dw = model(xf, zf)
    print([x.shape for x in [x, cls, cls_f, reg_dw]])
