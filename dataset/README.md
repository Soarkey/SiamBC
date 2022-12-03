# Testing Dataset Structure

> All json files download path: 
> https://github.com/StrangerZhang/pysot-toolkit#download-dataset

- [x] OTB100
    > official site: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
    >
    > download shell: https://github.com/foolwood/DaSiamRPN/blob/master/code/data/get_otb_data.sh
    ```shell
    dataset/
    └── OTB100/
       |── Basketball
       |── Biker
       └── (...)
    ```
  
- [x] UAV123
    > official site：https://uav123.org/
    > 
    > hyper.ai download link（support aria2c/magnet/https）： https://hyper.ai/datasets/5154
    >
    > Note: :bulb: `Dataset_UAV123.zip` is adopted as testing dataset.
    
    ```shell
    dataset/
    └── UAV123/
       |── bike1
       |── bike2
       └── (...)
    ```
  
- [x] VOT2019
    > download shell: https://github.com/jvlmdr/trackdat/blob/master/scripts/download_vot.sh
    
    ```shell
    dataset/
    └── VOT2019/
       |── agility
       |── ants1
       └── (...)
    ```
  
- [x] GOT10K
  
    > test on local & evaluate on server,
    > 
    > see submit instruction: http://got-10k.aitestunion.com/submit_instructions  

    ```shell
    dataset/
    └── GOT10K/
       |── GOT-10k_Test_000001/
       |── GOT-10k_Test_000002/
       └── (...)
    ```

- [x] LaSOT
  > download follow: http://vision.cs.stonybrook.edu/~lasot/download.html

  ```shell
  dataset/
  └── LASOT/
     |── airplane-1/
     |── airplane-13/
     |── airplane-15/
     └── (...)
  ```

- [x] TrackingNet
  > download follow: https://github.com/SilvioGiancola/TrackingNet-devkit
  
  ```shell
  dataset/
  └── TrackingNet/
     └── TEST/
        ├── anno/
        ├── frames/
        └── zips/
  ```
