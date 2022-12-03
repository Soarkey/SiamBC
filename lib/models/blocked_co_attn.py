# -*- encoding: utf-8 -*-

# @Author : soarkey
# @Date   : 2022/3/7
# @Desc   : block-based cooperate attention module

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import config
from lib.models.dcn import DeformConvPack


# channel attention
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        """
        Args:
            x: [B,256,7,7]
        Returns:
            c_weight: [B,256,1,1]
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale


# block-based cooperate attention
class BCA(nn.Module):
    def __init__(self, in_channel=256):
        super(BCA, self).__init__()

        self.in_channel = in_channel
        self.hidden_channel = in_channel // 2

        # feature extraction
        if config.TRAIN.ATTN.BLOCK_FEATURE:
            # template branch
            self.g1 = nn.Sequential(
                DeformConvPack(in_channel, self.hidden_channel, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(self.hidden_channel),
                nn.ReLU(inplace=True),
                DeformConvPack(self.hidden_channel, self.hidden_channel, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(self.hidden_channel),
                nn.ReLU(inplace=True),
            )

            # search branch
            self.g2 = nn.Sequential(
                DeformConvPack(in_channel, self.hidden_channel, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(self.hidden_channel),
                nn.ReLU(inplace=True),
            )

            self.theta = DeformConvPack(in_channel, self.hidden_channel, kernel_size=3, stride=2, padding=0)
            self.phi = DeformConvPack(in_channel, self.hidden_channel, kernel_size=3, stride=2, padding=0)

            self.Q = nn.Sequential(
                DeformConvPack(self.hidden_channel, 2 * in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2 * in_channel),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.UpsamplingBilinear2d(size=(7, 7)),
                DeformConvPack(self.hidden_channel, in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            )

            self.K1 = nn.Sequential(
                DeformConvPack(self.hidden_channel, 2 * in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2 * in_channel),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            )

            self.K2 = nn.Sequential(
                DeformConvPack(self.hidden_channel, in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            )

            self.alpha = nn.Parameter(1 * torch.ones(1).cuda())
            self.beta = nn.Parameter(1 * torch.ones(1).cuda())

        if config.TRAIN.ATTN.CHANNEL_ATTN:
            # channel attention
            self.ChannelGate = ChannelGate(in_channel)

    def forward(self, z, x):
        """
        args:
            z: [B,256,7,7]
            x: [B,256,31,31]
        returns:
            z: [B,256,7,7]
            x: [B,256,31,31]
        """

        b, _, z_height, z_width = z.shape
        _, _, x_height, x_width = x.shape

        if config.TRAIN.ATTN.BLOCK_FEATURE:
            zh, zw, xh, xw = [(x - 2) // 2 + 1 for x in [z_height, z_width, x_height, x_width]]

            zg = self.g1(z).view(b, self.hidden_channel, -1)
            zg = zg.permute(0, 2, 1).contiguous()

            xg = self.g2(x).view(b, self.hidden_channel, -1)
            xg = xg.permute(0, 2, 1).contiguous()

            z_theta = self.theta(z).view(b, self.hidden_channel, -1)
            z_theta = z_theta.permute(0, 2, 1)

            x_phi = self.phi(x).view(b, self.hidden_channel, -1)

            f = torch.matmul(z_theta, x_phi)  # [32,25,64]*[32,64,625]=[32,25,625]
            n = f.size(-1)
            fz_div_c = f / n

            f = f.permute(0, 2, 1).contiguous()
            n = f.size(-1)
            fx_div_c = f / n

            zn = torch.matmul(fz_div_c, xg)
            zn = zn.permute(0, 2, 1).contiguous().view(b, self.hidden_channel, zh, zw)
            zn = self.Q(zn)

            zq = self.alpha * zn + z

            xn = torch.matmul(fx_div_c, zg)
            xn = xn.permute(0, 2, 1).contiguous().view(b, self.hidden_channel, xh, xw)
            xn = self.K2(F.upsample_bilinear(self.K1(xn), size=(x_width, x_height)))  # [32,256,31,31]

            xk = self.beta * xn + x
        else:
            zq = z
            xk = x

        # response in channel weight
        if config.TRAIN.ATTN.CHANNEL_ATTN:
            # channel attention
            c_weight = self.ChannelGate(zq)
            zq = zq * c_weight
            xk = xk * c_weight

        return zq, xk


if __name__ == '__main__':
    kernel = torch.randn([32, 256, 7, 7]).cuda()
    search = torch.randn([32, 256, 31, 31]).cuda()

    # params & latency
    from thop.profile import profile
    from thop.utils import clever_format
    import time


    def calc_macs_and_params(name, model, kernel, search):
        model = model.cuda()
        macs, params = profile(model, inputs=(kernel, search), verbose=False)
        macs, params = clever_format([macs, params], "%3.3f")

        # test latency
        T_w = 20
        T_t = 100
        with torch.no_grad():
            for i in range(T_w):
                _, _ = model(kernel, search)
            start = time.time()
            for i in range(T_t):
                _, _ = model(kernel, search)
            end = time.time()
            latency = ((end - start) / T_t) * 1000

        print(f"{name.ljust(25)} | macs {macs} | params {params} | latency {latency}")


    calc_macs_and_params('bca', BCA(), kernel, search)
