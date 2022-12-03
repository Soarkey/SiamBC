# SiamBC

[SiamBC: Context-Related Siamese Network for Visual Object Tracking](https://doi.org/10.1109/access.2022.3192466)

<p align="center">
  <img width="85%" src="https://github.com/Soarkey/SiamBC/blob/master/assets/arch.png" alt="SiamBC"/>
</p>

## Abstract
The existing Siamese trackers have achieved increasingly results in visual tracking. However, the contextual association between template and search region is not been fully studied in previous Siamese Network-based methods, meanwhile, the feature information of the cross-correlation layer is investigated insufficiently. In this paper, we propose a new context-related Siamese network called SiamBC to address these issues. By introducing a cooperate attention mechanism based on block deformable convolution sampling features, the tracker can pre-match and enhance similar features to improve accuracy and robustness when the context embedding between the template and search fully interacted. In addition, we design a cascade cross correlation module. The cross-correlation layer of the stacked structure can gradually refine the deep information of the mined features and further improve the accuracy. Extensive experiments demonstrate the effectiveness of our tracker on six tracking benchmarks including OTB100, VOT2019, GOT10k, LaSOT, TrackingNet and UAV123. The code will be available at https://github.com/Soarkey/SiamBC.

## Installation

1. Prerequisites
    - linux / gcc 5+
    - python 3.6+ / pytorch 1.4+
    - CUDA 9.2+

2. Install python packages
    `pip install -r requirements.txt`

4. Build extensions (before test & eval on VOT dataset)
    > see https://github.com/StrangerZhang/pysot-toolkit
  
    ```shell
    python setup.py build_ext --inplace
    ```

## Pretrained model

-  form [SiamBAN](https://github.com/hqucv/siamban/blob/master/TRAIN.md#download-pretrained-backbones)
    - Google Drive: https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9

- from [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR#download-pretrained-backbones)
  - Baidu Pan(resnet50.model): https://pan.baidu.com/s/1IfZoxZNynPdY2UJ_--ZG2w (code: 7n7d)
  - Google Drive(resnet50.model & alexnet-bn.pth): https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9


## Dataset
```shell
.
├── GOT10K
├── GOT10K.json
├── LASOT
├── LASOT.json
├── NFS30
├── NFS30.json
├── OTB100
├── OTB100.json
├── TrackingNet
├── UAV123
├── UAV123.json
├── VOT2019
└── VOT2019.json
```


## Train
```shell
# cd experiments folder
cd /<path>/SiamBC/experiments/<experiment>/

# run train command
nohup python /data/SiamBC/tracking/train.py \
  --cfg train.yaml >> train.log 2>&1 &

# watch train log
tail train.log -f
```

## Test

- OTB100

```shell
nohup mpiexec --allow-run-as-root -n 6 \
python /data/SiamBC/tracking/test_epochs.py \
--start_epoch 30 \
--end_epoch 50 \
--gpu_nums 4 \
--threads 1 \
--dataset OTB100 \
--align True \
--type all >>test_otb.log 2>&1 &

tail test_otb.log -f
```

## Eval
```shell
nohup python /data/SiamBC/lib/eval_toolkit/bin/eval.py \
--dataset_dir ./dataset/OTB100 \
--dataset OTB100 \
--tracker_result_dir ./result/OTB100 \
--trackers 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 \
>eval_otb.log 2>&1 &

tail eval_otb.log -f
```

## Citation
If our work is useful for your research, please consider cite:
```tex  
@ARTICLE{SiamBC,
  author={He, Xiangwen and Sun, Yan},
  journal={IEEE Access},
  title={SiamBC: Context-Related Siamese Network for Visual Object Tracking},
  year={2022},
  volume={10},
  pages={76998-77010},
  doi={10.1109/ACCESS.2022.3192466}
}
```
