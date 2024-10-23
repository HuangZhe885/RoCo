# RoCo  （MM2024 Oral）
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
# Checkpoints and Results
* [RoCo_DAIR_V2X](https://drive.google.com/drive/folders/12alJjt4O_0SB3vTrosz7GmrWMqM9qLn7)

* [ RoCo_V2XSet](https://drive.google.com/drive/folders/1iy_T6EZ-s5VcNK-5tJ81TsORHdk1165F)

Download them and save them to  `opencood/logs`

#  How to use
1.  We are improving our project platform based on CoAlign. You just need to replace the `box_align_v2.py` and `intermedia_fusion_dataset.py` files.

2.  If you want to visualize the pose error, use `evaluate_pose_graph.py` in the `tool` folder.
3.  Important: During the graph matching and optimization process, the parameter `candidate_radius` needs to be adjusted according to different datasets. For specific parameter details, refer to the experiments in RoCo. https://github.com/HuangZhe885/RoCo/blob/bf9747b394fd018a8f0f3c2a3a5af6f71fadcd74/models/sub_modules/box_align_v2.py#L449
4.  The [bounding boxes](https://drive.google.com/drive/folders/1otDzESlepuhRBE4ZgJQfpArnpG1TG8uu) used in RoCo also come from saved files. You can download and save to `opencood/logs`,

# Citation
```
@inproceedings{huang2024roco,
  title={RoCo: Robust Cooperative Perception By Iterative Object Matching and Pose Adjustment},
  author={Huang, Zhe and Wang, Shuo and Wang, Yongcai and Li, Wanting and Li, Deying and Wang, Lei},
  booktitle={ACM Multimedia 2024}
}
```
#  Acknowlege
This project is impossible without the code of OpenCOOD, g2opy and d3d.

Thanks to [@DerrickXuNu](https://github.com/DerrickXuNu) and [@yifanlu0227](https://github.com/yifanlu0227)  for the great code framework.

Once again, my sincere thanks to [@yifanlu0227](https://github.com/yifanlu0227)  for his patient and meticulous help.

#  Video

https://github.com/user-attachments/assets/b11415ce-c284-4db4-a850-1eafd240d652











