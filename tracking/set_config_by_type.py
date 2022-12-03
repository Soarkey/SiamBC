# -*- encoding: utf-8 -*-

"""
@Author : Soarkey
@Date   : 2021/10/19
@Desc   : set model configs by different network architecture type
"""

from lib.core.config import config


def set_config_by_type(type):
    print("setting type: ", type)

    if type == 'base':
        config.TRAIN.LABEL_ASSIGN = False
    elif type == 'ca_cls':
        config.TRAIN.ATTN.TYPE = 'CA'
        config.TRAIN.ATTN.BRANCH = ['cls']
    elif type == 'ca_reg':
        config.TRAIN.ATTN.TYPE = 'CA'
        config.TRAIN.ATTN.BRANCH = ['reg']
    elif type == 'bca_cls':
        config.TRAIN.ATTN.TYPE = 'bca'
        config.TRAIN.ATTN.BRANCH = ['cls']
    elif type == 'bca_no_block_feature':
        config.TRAIN.ATTN.TYPE = 'bca'
        config.TRAIN.ATTN.BRANCH = ['cls']
        config.TRAIN.ATTN.BLOCK_FEATURE = False
    elif type == 'bca_no_chan_attn':
        config.TRAIN.ATTN.TYPE = 'bca'
        config.TRAIN.ATTN.BRANCH = ['cls']
        config.TRAIN.ATTN.CHANNEL_ATTN = False
    elif type == 'bca_reg':
        config.TRAIN.ATTN.TYPE = 'bca'
        config.TRAIN.ATTN.BRANCH = ['reg']
    elif type == 'bca_all':
        config.TRAIN.ATTN.TYPE = 'bca'
        config.TRAIN.ATTN.BRANCH = ['cls', 'reg']
    elif type == 'c_dwcorr':
        config.TRAIN.TOWER.TYPE = 'cascade_xcorr_tower'
        config.TRAIN.TOWER.STAGE = 1
    elif type == 'c2_dwcorr':
        config.TRAIN.TOWER.TYPE = 'cascade_tower'
        config.TRAIN.TOWER.STAGE = 2
    elif type == 'c3_dwcorr':
        config.TRAIN.TOWER.TYPE = 'cascade_tower'
        config.TRAIN.TOWER.STAGE = 3
    elif type == 'all':
        config.TRAIN.ATTN.TYPE = 'BCA'
        config.TRAIN.ATTN.BRANCH = ['cls', 'reg']
        config.TRAIN.ATTN.BLOCK_FEATURE = True
        config.TRAIN.ATTN.CHANNEL_ATTN = True
        config.TRAIN.TOWER.TYPE = 'cascade_tower'
        config.TRAIN.TOWER.STAGE = 1
