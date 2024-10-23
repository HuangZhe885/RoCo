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


# Here are some necessary installation steps 
1.  Create a conda environment
   All operations should be done on machines with GPUs.
   ```
   conda create -n RoCo python=3.8 pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
   conda activate RoCo 
   conda install git-lfs boost cuda cuda-nvcc mkl=2024.0 cmake -c conda-forge -c nvidia/label/cuda-11.6.2
   ```
2. Install spconv 1.2.1 or spconv 2.x
     Installing spconv 2.x is much more convenient.
    ```
    pip install spconv-cu116
   ```
3. if you want to use Spconv 1.2.1
```
# STEP 1: get source code. 

# do not clone it within the CoAlign repo
cd .. # just go somewhere else
git clone https://github.com/traveller59/spconv.git 
cd spconv
git checkout v1.2.1
git submodule update --init --recursive 

# STEP 2: compile
python setup.py bdist_wheel

# STEP 3: install
cd ./dist
pip install spconv-1.2.1-cp38-cp38m-linux_x86_64.whl

# check if is successfully installed
python 
import spconv
```
4. Install some other packages

```pip install -r requirements.txt```

6. Install RoCo
```
https://github.com/HuangZhe885/RoCo.git
cd RoCo
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
# FPVRCNN's iou_loss dependency (optional)
python opencood/pcdet_utils/setup.py build_ext --inplace 
```
# Training 训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch  --nproc_per_node=6 --use_env opencood/tools/train_ddp.py -y YAML_FILE [--model_dir MODEL_DIR]
```
* **-y YAML_FILE**  the yaml configuration file
* **[--model_dir MODEL_FOLDER]** is optional, indicating that training continues from this log (resume training). It will read config.yaml from under** MODEL_FOLDER** instead of the input **-y YAML_FILE**. so it can be written **-y None** and no yaml file is provided

# Testing 测试
```
python opencood/tools/inference.py --model_dir MODEL_DIR --fusion_method intermediate
```


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











