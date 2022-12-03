from __future__ import absolute_import

import numpy as np
from easydict import EasyDict as edict

from lib.tracker.siambc import SiamBC
from lib.utils.utils import cxy_wh_2_rect, get_axis_aligned_bbox
from . import Tracker


class IdentityTracker(Tracker):
    def __init__(self, name, net, dataset='OTB100', align=True):
        super(IdentityTracker, self).__init__(
            name=name,
            net=net,
            is_deterministic=True)
        self.siam_net = net
        # params
        siam_info = edict()
        siam_info.dataset = dataset
        siam_info.epoch_test = True
        siam_info.align = True if 'VOT' in dataset and align == 'True' else False
        self.siam_tracker = SiamBC(siam_info)

    def init(self, image, box):
        cx, cy, w, h = get_axis_aligned_bbox(box)

        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        self.state = self.siam_tracker.init(image, target_pos, target_sz, self.siam_net)
        self.box = box

    def update(self, image):
        self.state = self.siam_tracker.track(self.state, image)
        self.box = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
        return self.box
