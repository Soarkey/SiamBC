import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2,3"
config.WORKERS = 16
config.PRINT_FREQ = 40
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

config.TRAIN = edict()
config.TEST = edict()
config.DATASET = edict()
config.DATASET.VID = edict()
config.DATASET.GOT10K = edict()
config.DATASET.COCO = edict()
config.DATASET.DET = edict()
config.DATASET.LASOT = edict()
config.DATASET.YTB = edict()

# augmentation
config.DATASET.SHIFT = 4
config.DATASET.SCALE = 0.05
config.DATASET.COLOR = 1
config.DATASET.FLIP = 0
config.DATASET.BLUR = 0
config.DATASET.GRAY = 0
config.DATASET.MIXUP = 0
config.DATASET.CUTOUT = 0
config.DATASET.CHANNEL6 = 0
config.DATASET.LABELSMOOTH = 0
config.DATASET.ROTATION = 0
config.DATASET.SHIFTs = 64
config.DATASET.SCALEs = 0.18
config.DATASET.RANDOM_ERASE = 0

# vid
config.DATASET.VID.PATH = '/data/siamx/training_dataset/vid/crop511'
config.DATASET.VID.ANNOTATION = '/data/siamx/training_dataset/vid/train.json'
config.DATASET.VID.RANGE = 100
config.DATASET.VID.USE = 110000

# youtube-bb
config.DATASET.YTB.PATH = '/data/siamx/training_dataset/yt_bb/crop511'
config.DATASET.YTB.ANNOTATION = '/data/siamx/training_dataset/yt_bb/train.json'
config.DATASET.YTB.RANGE = 3
config.DATASET.YTB.USE = 210000

# got10k
config.DATASET.GOT10K.PATH = '/data/dataset/got_10k/crop511'
config.DATASET.GOT10K.ANNOTATION = '/data/dataset/got_10k/train.json'
config.DATASET.GOT10K.RANGE = 100
config.DATASET.GOT10K.USE = 200000

# det
config.DATASET.DET.PATH = '/data/siamx/training_dataset/det/crop511'
config.DATASET.DET.ANNOTATION = '/data/siamx/training_dataset/det/train.json'
config.DATASET.DET.RANGE = 100
config.DATASET.DET.USE = 60000

# coco
config.DATASET.COCO.PATH = '/data/siamx/training_dataset/coco/crop511'
config.DATASET.COCO.ANNOTATION = '/data/siamx/training_dataset/coco/train2017.json'
config.DATASET.COCO.RANGE = 1
config.DATASET.COCO.USE = 60000

# lasot
config.DATASET.LASOT.PATH = '/data/siamx/training_dataset/lasot/crop511'
config.DATASET.LASOT.ANNOTATION = '/data/siamx/training_dataset/lasot/train.json'
config.DATASET.LASOT.RANGE = 100
config.DATASET.LASOT.USE = 200000

# train
config.TRAIN.MODEL = 'SiamBC'
config.TRAIN.OPTIMIZER = 'SGD'
config.TRAIN.BACKBONE = 'ResNet'
config.TRAIN.LABEL_ASSIGN = False  # 同心圆标签
config.TRAIN.AUG_DATASET = False  # 使用自己修改后的增强数据方法

config.TRAIN.ATTN = edict()
config.TRAIN.ATTN.TYPE = None  # 注意力模块类型, 有 CA 和 bca
config.TRAIN.ATTN.BRANCH = ['cls']  # 注意力模块位置
config.TRAIN.ATTN.BLOCK_FEATURE = True  # 是否包含区块特征提取模块
config.TRAIN.ATTN.CHANNEL_ATTN = True  # 是否包含通道注意力模块

config.TRAIN.LABEL_SIZE = 25  # 标签大小, 默认25

# 损失权重
config.TRAIN.CLS_LOSS_WEIGHT = 1.0
config.TRAIN.REG_LOSS_WEIGHT = 1.0

config.TRAIN.USED_LAYERS = [3]
config.TRAIN.IN_CHANNELS = 1024
config.TRAIN.HIDDEN_CHANNELS = 256

config.TRAIN.TOWER = edict()
config.TRAIN.TOWER.TYPE = 'box_tower'  # 互相关模块， 有 box_tower, cascade_tower
config.TRAIN.TOWER.STAGE = 1  # 级联互相关模块 stage 层数

config.TRAIN.RESUME = False
config.TRAIN.START_EPOCH = 0
config.TRAIN.END_EPOCH = 50
config.TRAIN.TEMPLATE_SIZE = 127
config.TRAIN.SEARCH_SIZE = 255
config.TRAIN.STRIDE = 8
config.TRAIN.BATCH = 32
config.TRAIN.PRETRAIN = 'pretrain.model'
config.TRAIN.LR_POLICY = 'log'
config.TRAIN.LR = 0.001
config.TRAIN.LR_END = 0.00001
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WEIGHT_DECAY = 0.0001
config.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'

# test
config.TEST.DATA = 'VOT2019'
config.TEST.START_EPOCH = 30
config.TEST.END_EPOCH = 50

config.TEST.OTB100 = edict()
config.TEST.OTB100.penalty_k = 0.087
config.TEST.OTB100.lr = 0.408
config.TEST.OTB100.window_influence = 0.366
config.TEST.OTB100.small_sz = 271
config.TEST.OTB100.big_sz = 271

config.TEST.GOT10K = edict()
config.TEST.GOT10K.penalty_k = 0.022
config.TEST.GOT10K.lr = 0.799
config.TEST.GOT10K.window_influence = 0.118
config.TEST.GOT10K.small_sz = 255
config.TEST.GOT10K.big_sz = 255

config.TEST.VOT2019 = edict()
config.TEST.VOT2019.penalty_k = 0.062
config.TEST.VOT2019.lr = 0.765
config.TEST.VOT2019.window_influence = 0.380
config.TEST.VOT2019.small_sz = 255
config.TEST.VOT2019.big_sz = 271
config.TEST.VOT2019.ratio = 0.94

config.TEST.VOT2018 = edict()
config.TEST.VOT2018.penalty_k = 0.021
config.TEST.VOT2018.lr = 0.730
config.TEST.VOT2018.window_influence = 0.321
config.TEST.VOT2018.small_sz = 255
config.TEST.VOT2018.big_sz = 271
config.TEST.VOT2018.ratio = 0.93

config.TEST.LASOT = edict()
config.TEST.LASOT.penalty_k = 0.11
config.TEST.LASOT.lr = 0.7
config.TEST.LASOT.window_influence = 0.20
config.TEST.LASOT.small_sz = 255
config.TEST.LASOT.big_sz = 255


def _update_dict(k, v):
    if k in ['TRAIN', 'TEST']:
        for vk, vv in v.items():
            config[k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[k][vk][vvk] = vvv
                    except:
                        config[k][vk] = edict()
                        config[k][vk][vvk] = vvv

    else:
        config[k] = v  # gpu et.


def update_config(config_file):
    with open(config_file) as f:
        model_config = edict(yaml.safe_load(f))
        for k, v in model_config.items():
            if k in config:
                _update_dict(k, v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
