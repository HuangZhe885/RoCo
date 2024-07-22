# RoCo  （MM2024）
Robust Cooperative Perception By Iterative Object Matching and Pose Adjustment


![WeChat7f6b3acd42cbba3c9648bcf1fc13b3c9](https://github.com/user-attachments/assets/5674d1f6-5b57-4f09-b129-151f4130739f)

# Installation
You can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare data and install RoCo. The installation is totally the same as [CoAlign](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie).

# Data Preparation

mkdir a dataset folder under RoCo. Put your OPV2V, V2XSet, DAIR-V2X data in this folder. You just need to put in the dataset you want to use.
RoCo/dataset. All data configurations are the same as [CoAlign](https://github.com/yifanlu0227/CoAlign?tab=readme-ov-file). For details, please refer to CoAlign.

```
├── my_dair_v2x 
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
├── V2XSET
│   ├── test
│   ├── train
│   └── validate
```
#  How to use
(1) We are improving our project platform based on CoAlign. You just need to replace the `box_align_v2.py` and `intermedia_fusion_dataset.py` files.

(2) If you want to visualize the pose error, use `evaluate_pose_graph.py` in the `tool` folder.

#  Acknowlege
This project is impossible without the code of OpenCOOD, g2opy and d3d.

Thanks to [@DerrickXuNu](https://github.com/DerrickXuNu) and [@yifanlu0227](https://github.com/yifanlu0227)  for the great code framework.

Once again, my sincere thanks to [@yifanlu0227](https://github.com/yifanlu0227)  for his patient and meticulous help.









