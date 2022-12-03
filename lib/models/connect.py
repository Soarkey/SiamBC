import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import config


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        feature = self.head(feature)
        return feature


class MultiDiCorr(nn.Module):
    """
    For tensorRT version
    """

    def __init__(self, inchannels=512, outchannels=256):
        super(MultiDiCorr, self).__init__()
        self.cls_encode = matrix(in_channels=inchannels, out_channels=outchannels, type='cls')
        self.reg_encode = matrix(in_channels=inchannels, out_channels=outchannels, type='reg')

    def forward(self, search, kernal):
        """
        :param search:
        :param kernal:
        :return:  for tensor2trt
        """
        cls_z0, cls_z1, cls_z2, cls_x0, cls_x1, cls_x2 = self.cls_encode(kernal, search)  # [z11, z12, z13]
        reg_z0, reg_z1, reg_z2, reg_x0, reg_x1, reg_x2 = self.reg_encode(kernal, search)  # [x11, x12, x13]

        return cls_z0, cls_z1, cls_z2, cls_x0, cls_x1, cls_x2, reg_z0, reg_z1, reg_z2, reg_x0, reg_x1, reg_x2


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, crop=False):
        """
        Args:
            x: [B,1024,15,15]
            crop:
        Returns:
        """
        x_ori = self.downsample(x)  # x_ori [B,256,15,15]
        if x_ori.size(3) < 20 and crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]

        if not crop:
            return x_ori
        else:
            return x_ori, xf


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(AdjustAllLayer, self).__init__()
        self.num = len(in_channels)
        for i in range(self.num):
            self.add_module('downsample' + str(i + 2), AdjustLayer(in_channels[i], out_channel))

    def forward(self, features, crop=False):
        """
        Args:
            features: list[3]->layer2,3,4
            crop:
        Returns:
            cls_f, reg_f
        """
        if crop:
            down_layer = getattr(self, 'downsample3')
            out = down_layer(features[1])
            if out.size(3) < 20:
                l = 4
                r = -4
                out = out[:, :, l:r, l:r]
            return out

        # 针对分类分支, 使用三层融合,
        # 针对回归分支, 只使用L4特征
        out = []
        for i in range(self.num):
            down_layer = getattr(self, 'downsample' + str(i + 2))
            out.append(down_layer(features[i]))
            # if out[i].size(3) < 20 and crop:
            #     l = 4
            #     r = -4
            #     out[i] = out[i][:, :, l:r, l:r]

        reg_f = out[1].clone()

        return out[1], reg_f


class AlignedModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.down_l = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.flow_make = nn.Conv2d(out_channel * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, low_feature, h_feature):
        """
        Args:
            low_feature: [B,C,H,W]
            h_feature:   [B,C,H,W]
        Returns:
        """
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


# --------------------
# Ocean module
# --------------------
class matrix(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels, out_channels, type='cls'):
        super(matrix, self).__init__()

        # same size (11)
        self.matrix11_k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # h/2, w
        self.matrix12_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix12_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # w/2, h
        self.matrix21_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix21_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.type = type
        if config.TRAIN.ATTN.TYPE is not None \
            and config.TRAIN.ATTN.TYPE == 'bca' \
            and config.TRAIN.TOWER.TYPE == 'box_tower' \
            and self.type in config.TRAIN.ATTN.BRANCH:

            from lib.models.blocked_co_attn import BCA
            self.attn = BCA(out_channels)

    def forward(self, z, x):
        # z [32,C,15,15]
        # x [32,C,31,31]
        if config.TRAIN.ATTN.TYPE is not None:
            if config.TRAIN.TOWER.TYPE == 'box_tower' and self.type in config.TRAIN.ATTN.BRANCH:
                z, x = self.attn(z, x)

        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)

        z12 = self.matrix12_k(z)
        x12 = self.matrix12_s(x)

        z21 = self.matrix21_k(z)
        x21 = self.matrix21_s(x)

        # z11 [32,256,5,5] x11 [32,256,29,29]
        # z12 [32,256,3,5] x12 [32,256,27,29]
        # z21 [32,256,5,3] x21 [32,256,29,27]

        return [z11, z12, z21], [x11, x12, x21]


class AdaptiveConv(nn.Module):
    """ Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors"""

    def __init__(self, in_channels, out_channels):
        super(AdaptiveConv, self).__init__()
        self.conv = DeformConv(in_channels, out_channels, 3, padding=1)

    def forward(self, x, offset):
        N, _, H, W = x.shape
        assert offset is not None
        assert H * W == offset.shape[1]
        # reshape [N, NA, 18] to (N, 18, H, W)
        offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
        x = self.conv(x, offset)

        return x


class AlignHead(nn.Module):
    # align features and classification score

    def __init__(self, in_channels, feat_channels):
        super(AlignHead, self).__init__()

        self.rpn_conv = AdaptiveConv(in_channels, feat_channels)
        self.rpn_cls = nn.Conv2d(feat_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, offset):
        x = self.relu(self.rpn_conv(x, offset))
        cls_score = self.rpn_cls(x)
        return cls_score


class GroupDW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self):
        super(GroupDW, self).__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, z, x):
        z11, z12, z21 = z
        x11, x12, x21 = x

        re11 = xcorr_depthwise(x11, z11)
        re12 = xcorr_depthwise(x12, z12)
        re21 = xcorr_depthwise(x21, z21)
        re = [re11, re12, re21]

        # weight
        weight = F.softmax(self.weight, 0)

        s = 0
        for i in range(3):
            s += weight[i] * re[i]

        return s


class SingleDW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels=256):
        super(SingleDW, self).__init__()

    def forward(self, z, x):
        s = xcorr_depthwise(x, z)

        return s


class box_tower(nn.Module):
    """
    box tower for FCOS reg
    """

    def __init__(self, inchannels=512, outchannels=256, towernum=1):
        super(box_tower, self).__init__()
        tower = []
        cls_tower = []
        # encode backbone
        self.cls_encode = matrix(in_channels=inchannels, out_channels=outchannels, type='cls')
        self.reg_encode = matrix(in_channels=inchannels, out_channels=outchannels, type='reg')
        self.cls_dw = GroupDW()
        self.reg_dw = GroupDW()

        # box pred head
        for i in range(towernum):
            tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        # cls tower
        for i in range(towernum):
            cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, kernel, update=None):
        # encode first
        if update is None:
            cls_z, cls_x = self.cls_encode(kernel, search)  # [z11, z12, z13]
        else:
            cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]

        reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]

        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)  # [32,256,25,25]
        x_reg = self.bbox_tower(reg_dw)  # [32,256,25,25]
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)

        # cls tower
        c = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(c)
        # x[32,4,25,25] cls[32,1,25,25] cls_dw[32,256,25,25] x_reg[32,256,25,25]
        return x, cls, cls_dw, x_reg
