# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import seaborn as sns
import json
import copy
import numpy as np
from opencood.models.sub_modules.box_align_v2 import vis_pose_graph, box_alignment_relative_sample_np
from opencood.utils.pose_utils import generate_noise
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib import rcParams
# sns.set(rc={'figure.figsize':(11.7,8.27)})
from scipy.stats import expon

from scipy.stats import norm
#------4.1 zh update

import matplotlib.pyplot as plt
import numpy as np

# def vis_trans(trans_error_list,filenames,save_path, std):
#       # 
#   plt.figure()
#   plt.clf()
#   for data in [trans_error_list[0], trans_error_list[2]]: 
#     param = expon.fit(data) 
#     x = np.linspace(0.01, max(data), 100) 
#     y = expon.pdf(x, *param) 

#     plt.legend(labels=[filenames[0], filenames[2]])
#     plt.plot(x, y, linewidth=2)

#   plt.xlabel('trans error(m)')
#   plt.ylabel('Denisty')
#   plt.legend(labels=[filenames[0],filenames[2]])
#   plt.xlim(xmin=0,xmax=6.0) 
#   plt.show()
#   plt_filename = f"{std}_".replace(".","") + "trans_new.png"
#   plt_filename = os.path.join(save_path, plt_filename)
#   plt.savefig(plt_filename, dpi=300)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

def vis_trans(trans_error_list, filenames, save_path, std):
    # 
    plt.figure()
    plt.clf()

    # 
    for i, data in enumerate([trans_error_list[0], trans_error_list[2]]):
        param = expon.fit(data)
        x = np.linspace(0.01, max(data), 100)
        y = expon.pdf(x, *param)
        
        # 
        linestyle = '-' if i == 0 else '--'
        
        # 
        plt.plot(x, y, linewidth=2, linestyle=linestyle, label=filenames[i])

    plt.legend()

    plt.xlabel('trans error(m)')

    plt.ylabel('Density')

    plt.xlim(xmin=0, xmax=6.0)

    plt.show()


    plt_filename = f"{std}_".replace(".", "") + "trans_new.png"
    plt_filename = os.path.join(save_path, plt_filename)
    plt.savefig(plt_filename, dpi=300)

def vis_rot(rot_error_list, filenames, save_path, std):
    plt.figure()
    plt.clf()


    for i, data in enumerate([rot_error_list[0], rot_error_list[2]]):
        param = expon.fit(data)
        x = np.linspace(0.01, max(data), 100)
        y = expon.pdf(x, *param)
        linestyle = '-' if i == 0 else '--'

        plt.plot(x, y, linewidth=2, linestyle=linestyle, label=filenames[i])

    plt.legend()
    plt.xlabel('rot error(°)')
    plt.ylabel('Density')
    plt.xlim(xmin=0, xmax=6)
    plt.show()
    plt_filename = f"{std}_".replace(".", "") + "rot_new.png"
    plt_filename = os.path.join(save_path, plt_filename)
    plt.savefig(plt_filename, dpi=300)
  





DEBUG = True

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


            

def evaluate_pose_graph(data_dict, save_path, std=0.8):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    filenames = ['w/o matching', 'with matching', 'with matching']

    test_term_num = len(filenames)

    trans_error_list = [[] for i in range(test_term_num)]
    
    rot_error_list = [[] for i in range(test_term_num)]


    np.random.seed(100)

    for sample_idx, content in tqdm(data_dict.items()): 
        print(sample_idx)
        if content is None:
            continue
        pred_corners_list = content['pred_corner3d_np_list']
        pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
        uncertainty_list = content['uncertainty_np_list']
        uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
        lidar_pose_clean_np = np.array(content['lidar_pose_clean_np'], dtype=np.float64)
        lidar_pose_clean_dof3 = lidar_pose_clean_np[:,[0,1,4]]
        cav_id_list = content['cav_id_list']
        N = lidar_pose_clean_np.shape[0]

        noisy_lidar_pose = copy.deepcopy(lidar_pose_clean_np)
        noisy_lidar_pose[1:,[0,1,4]] += np.random.normal(0, std, size=(N-1,3))
        noisy_lidar_pose_dof3 = noisy_lidar_pose[:,[0,1,4]]


        pose_after = [
                      box_alignment_relative_sample_np(pred_corners_list, # l2_cd_1.5
                                                        noisy_lidar_pose, 
                                                        uncertainty_list=uncertainty_list, 
                                                        landmark_SE2=True,
                                                        adaptive_landmark=False,
                                                        normalize_uncertainty=False,
                                                        abandon_hard_cases=True,
                                                        drop_hard_boxes=True,
                                                        use_uncertainty=True),

                      box_alignment_relative_sample_np(pred_corners_list, # l2_1.5
                                                        noisy_lidar_pose, 
                                                        uncertainty_list=uncertainty_list, 
                                                        landmark_SE2=True,
                                                        adaptive_landmark=False,
                                                        normalize_uncertainty=False,
                                                        abandon_hard_cases=True,
                                                        drop_hard_boxes=True,
                                                        use_uncertainty=False),

                      noisy_lidar_pose_dof3,
                    ]

        diffs = [np.abs(lidar_pose_clean_dof3 - pose) for pose in pose_after]
        diffs = np.stack(diffs)
        diffs[:,1:,2] = np.minimum(diffs[:,1:,2], 360 - diffs[:,1:,2]) # do not include ego
        
        for i, diff in enumerate(diffs):
            pos_diff = diff[1:,:2] # do not include ego
            angle_diff = diff[1:,2] # do not include ego
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            trans_error_list[i].extend(pos_diff.flatten().tolist())
            rot_error_list[i].extend(angle_diff.flatten().tolist())

        DEBUG = False
        if DEBUG:
            if (diffs[0] > 1).any():
                pose_graph_save_dir = os.path.join(save_path, f"pg_vis/{sample_idx}")
                vis_pose_graph(pose_after,
                            pred_corners_list, 
                            save_dir_path=pose_graph_save_dir)
                np.savetxt(os.path.join(pose_graph_save_dir,"pose_info.txt"), diffs.reshape(-1,3), fmt="%.4f")

        

    vis_trans(trans_error_list,filenames,save_path,std)   
    vis_rot(rot_error_list,filenames,save_path,std)
    # calc_data(trans_error_list, rot_error_list, filenames, save_path, std)
    
evaluate_json = "/home/hz/code/opencood/logs/v2xset/test/stage1_boxes.json"
data_dict = read_json(evaluate_json)


output_path = "/home/hz/code/vis_result"
evaluate_pose_graph(data_dict, output_path, std=0.8)
