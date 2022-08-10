# SiamBC

The code is being prepared and we will upload the source code once completed.

[SiamBC: Context-Related Siamese Network for Visual Object Tracking](https://doi.org/10.1109/access.2022.3192466)

<p align="center">
  <img width="85%" src="https://github.com/Soarkey/SiamBC/blob/main/assets/arch.png" alt="SiamBC"/>
</p>

# Abstract
The existing Siamese trackers have achieved increasingly results in visual tracking. However, the contextual association between template and search region is not been fully studied in previous Siamese Network-based methods, meanwhile, the feature information of the cross-correlation layer is investigated insufficiently. In this paper, we propose a new context-related Siamese network called SiamBC to address these issues. By introducing a cooperate attention mechanism based on block deformable convolution sampling features, the tracker can pre-match and enhance similar features to improve accuracy and robustness when the context embedding between the template and search fully interacted. In addition, we design a cascade cross correlation module. The cross-correlation layer of the stacked structure can gradually refine the deep information of the mined features and further improve the accuracy. Extensive experiments demonstrate the effectiveness of our tracker on six tracking benchmarks including OTB100, VOT2019, GOT10k, LaSOT, TrackingNet and UAV123. The code will be available at https://github.com/Soarkey/SiamBC.

# Citation
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
