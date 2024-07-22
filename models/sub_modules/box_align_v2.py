"""
Graph matching and optimization
"""


from opencood.models.sub_modules.pose_graph_optim import PoseGraphOptimization2D
from opencood.utils.transformation_utils import pose_to_tfm
from opencood.utils.common_utils import check_torch_to_numpy
from opencood.utils import box_utils
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import g2o
from icecream import ic
import copy
import os
import matplotlib.pyplot as plt
import math
import networkx as nx

DEBUG = False

def vis_pose_graph(poses, pred_corner3d, save_dir_path, vis_agent=False):
    """
    Args:
        poses: list of np.ndarray
            each item is a pose . [pose_before, ..., pose_refined]

        pred_corner3d: list
            predicted box for each agent.

        vis_agent: bool
            whether draw the agent's box

    """
    COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
    from opencood.utils.transformation_utils import get_relative_transformation

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for iter, pose in enumerate(poses):
        box_idx = 0
        # we first transform other agents' box to ego agent's coordinate
        relative_t_matrix = get_relative_transformation(pose)
        N = pose.shape[0]
        nonempty_indices = [idx for (idx, corners) in enumerate(pred_corner3d) if len(corners)!=0]
        pred_corners3d_in_ego = [box_utils.project_box3d(pred_corner3d[i], relative_t_matrix[i]) for i in nonempty_indices]

        for agent_id in range(len(pred_corners3d_in_ego)):
            if agent_id not in nonempty_indices:
                continue
            corner3d = pred_corners3d_in_ego[agent_id]
            agent_pos = relative_t_matrix[agent_id][:2,3] # agent's position in ego's coordinate

            if vis_agent:
                plt.scatter(agent_pos[0], agent_pos[1], s=4, c=COLOR[agent_id])

            corner2d = corner3d[:,:4,:2]
            center2d = np.mean(corner2d, axis=1)
            for i in range(corner2d.shape[0]):
                plt.scatter(corner2d[i,[0,1],0], corner2d[i,[0,1], 1], s=2, c=COLOR[agent_id])
                plt.plot(corner2d[i,[0,1,2,3,0],0], corner2d[i,[0,1,2,3,0], 1], linewidth=1, c=COLOR[agent_id])
                plt.text(corner2d[i,0,0], corner2d[i,0,1], s=str(box_idx), fontsize="xx-small")
                # add a line connecting box center and agent.
                box_center = center2d[i] # [2,]
                connection_x = [agent_pos[0], box_center[0]]
                connection_y = [agent_pos[1], box_center[1]]

                plt.plot(connection_x, connection_y,'--', linewidth=0.5, c=COLOR[agent_id], alpha=0.3)
                box_idx += 1
        
        filename = os.path.join(save_dir_path, f"{iter}.png")
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.savefig(filename, dpi=400)
        plt.clf()


def all_pair_l2(A, B):
    """ All pair L2 distance for A and B
    Args:
        A : np.ndarray
            shape [N_A, D]
        B : np.ndarray
            shape [N_B, D]
    Returns:
        C : np.ndarray
            shape [N_A, N_B]
    """
    TwoAB = 2*A@B.T  # [N_A, N_B]
    C = np.sqrt(
              np.sum(A * A, 1, keepdims=True).repeat(TwoAB.shape[1], axis=1) \
            + np.sum(B * B, 1, keepdims=True).T.repeat(TwoAB.shape[0], axis=0) \
            - TwoAB
        )
    return C


def compute_edge(a1, a2, b1, b2, pred_center_world_cat):
    """
    Args:
        a1: int
            agent id
        a2: int
            agent id
        b1: int
            box id
        b2: int
            box id
        pred_center_world_cat: np.ndarray
            shape [N, 3]
    Returns:
        edge: np.ndarray
            shape [3,]
    """
    edge_diff = pred_center_world_cat[a1] + pred_center_world_cat[b2] - pred_center_world_cat[a2] - pred_center_world_cat[b1]
    return edge_diff


def compute_graph_similarity_only_edge(idx, idx_candidate, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat):
    agent_idx = box_idx_to_agent[idx] 
    agent_candidate = box_idx_to_agent[idx_candidate]  

    begin_idx = sum(pred_len[:agent_idx])
    end_idx = begin_idx + pred_len[agent_idx]
    graph_idx = []
    graph_candidate = []
    for i in range(begin_idx, end_idx):
        if box_idx_to_agent[i] == agent_idx and idx != i:  
            if len(pre_candidate_set[i]) >= 1:  
                count = 0
                for j in pre_candidate_set[i]:
                    if box_idx_to_agent[j] == agent_candidate:
                        count += 1
                    
                if count == 1: 
                    graph_idx.append(i)
                    graph_candidate.append(j)

    score = 0
    for i in range(len(graph_idx)):
        edge_diff = compute_edge(idx, graph_idx[i], idx_candidate, graph_candidate[i], pred_center_world_cat) # / len(graph_candidate[i])
        score += math.exp(-np.linalg.norm(edge_diff))

    if len(graph_idx) > 0:
        score /= len(graph_idx)
        # print("graph score:", score)

    return score

def compute_graph_similarity_only_distance(idx, idx_candidate, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat):

    distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
    score = distance_score
    return score


def compute_graph_similarity(idx, idx_candidate, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat):
    """
    Args:
        idx: int
            the center of graph
        idx_candidate: int
            the center of graph
        pre_candidate_set: list
            each item is a list of box idx, which is the candidate list of the corresponding box
        pred_len: list
            each item is the number of boxes for each agent
        box_idx_to_agent: list
            each item is the agent id for each box
        pred_center_world_cat: np.ndarray
            shape [N, 3]
    Returns:
        similarity: float
            the similarity between two graphs
    """
    agent_idx = box_idx_to_agent[idx]  
    agent_candidate = box_idx_to_agent[idx_candidate]  

   
    begin_idx = sum(pred_len[:agent_idx])
    end_idx = begin_idx + pred_len[agent_idx]
    graph_idx = []
    graph_candidate = []
    for i in range(begin_idx, end_idx):
        if box_idx_to_agent[i] == agent_idx and idx != i:  
            if len(pre_candidate_set[i]) >= 1:  
                
                count = 0
                for j in pre_candidate_set[i]:
                    if box_idx_to_agent[j] == agent_candidate:
                        count += 1
                    
                if count == 1:  
                    graph_idx.append(i)
                    graph_candidate.append(j)
    
    if (len(graph_idx) == 0): 
        print("--------------no right graph----------------")
        print("--------------relax the thera----------------")
       

       
        # if pred_len[agent_idx] <= 1:
        #     return -1
        # #
        # limits = 1
        # while(len(graph_idx) <= 1):
        #     limits += 1
        #     for i in range(begin_idx, end_idx):
        #         if box_idx_to_agent[i] == agent_idx and idx != i:  
        #             if len(pre_candidate_set[i]) >= 1 and i not in graph_idx:  
        #                 count = 0
        #                 for j in pre_candidate_set[i]:
        #                     if box_idx_to_agent[j] == agent_candidate:
        #                         count += 1
        #                 if count == limits:  
        #                     graph_idx.append(i)
        #                     graph_candidate.append(j)
        #     if (limits > len(pred_len)): 
        #         return -1


    score = 0
    for i in range(len(graph_idx)):
        edge_diff = compute_edge(idx, graph_idx[i], idx_candidate, graph_candidate[i], pred_center_world_cat) # / len(graph_candidate[i])
        score += math.exp(-np.linalg.norm(edge_diff))

    if len(graph_idx) > 0:
        score /= len(graph_idx)
        print("graph score:", score)

    if len(graph_idx) == 0:
        distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
        score = distance_score
        return score
    
    
    distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
    score += distance_score
   
    
    # print(idx, idx_candidate, score)
    
    return score

    
    
def weighted_bipartite_matching_hopcroft_karp(idx_set: set, candidate_idx_set: set, candidate_set: dict, candidate_set_score: dict):

    print("idx_set", idx_set)
    print("candidate_idx_set", candidate_idx_set)
    print("candidate_set", candidate_set)
    print("candidate_set_score", candidate_set_score)
  
    G = nx.Graph()


   
    left_nodes = list(idx_set)
    right_nodes = list(candidate_idx_set)
    # print("left_nodes",left_nodes)
    # print("right_nodes",right_nodes)

 
    G.add_nodes_from(left_nodes, bipartite=0)  
    G.add_nodes_from(right_nodes, bipartite=1)  

    # print("nodes ::", G.nodes())
    # print(candidate_set)
    # print(candidate_set_score)


    for i in range(len(left_nodes)):
        for j in range(len(candidate_set[left_nodes[i]])):
            G.add_edge(left_nodes[i], candidate_set[left_nodes[i]][j], weight=candidate_set_score[left_nodes[i]][j])

   
    # for i in range(left_nodes):
    #     for j in range(num_targets):
    #         G.add_edge(i, j + num_agents, weight=weights[i, j])  

   
    matching = nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes=left_nodes)


    # matched_pairs = [(agent, target - num_agents) for left, right in matching.items() if agent in agent_nodes]
    matched_pairs = [(left, right) for left, right in matching.items() ]

    matched_nodes = set(pair[0] for pair in matched_pairs)
    unmatched_nodes = set(left_nodes) - matched_nodes

    return matched_pairs, unmatched_nodes


def multi_weighted_bipartite_matching_hopcroft_karp(idx_set: set, candidate_idx_set: set, candidate_set: dict, candidate_set_score: dict, box_idx_to_agent: list):

    agent_number = len(set(box_idx_to_agent))
    if agent_number == 1:
        return
    elif agent_number == 2:
        return weighted_bipartite_matching_hopcroft_karp(idx_set, candidate_idx_set, candidate_set, candidate_set_score)


    

    return 

# use HK
def box_alignment_relative_sample_np(
            pred_corners_list,     
            noisy_lidar_pose,     
            uncertainty_list=None, 
            landmark_SE2=True,  
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False, 
            drop_unsure_edge = False, 
            use_uncertainty = True,
            thres = 1.5,    
            yaw_var_thres = 0.2,   
            max_iterations = 1000):  
    """ Perform box alignment for one sample. 
    Correcting the relative pose.

    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        clean_lidar_poses:
            [N_cav1, 6], in degree
        
        noisy_lidar_poses:
            [N_cav1, 6], in degree

        uncertainty_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

        landmark_SE2:
            if True, the landmark is SE(2), otherwise R^2
        
        adaptive_landmark: (when landmark_SE2 = True)
            if True, landmark will turn to R^2 if yaw angles differ a lot

        normalize_uncertainty: bool
            if True, normalize the uncertainty
        
        abandon_hard_cases: bool
            if True, algorithm will just return original poses for hard cases

        drop_unsure_edge: bool

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """
    print("noisy_lidar_pose_GT",noisy_lidar_pose)
    print("noisy_lidar_poseGT_SHAPE",noisy_lidar_pose.shape)
    

    # with open('/home/hz/code/opencood/logs/Other_results/noisy_lidar_pose_beforeAlign.txt', 'a') as f:
    #     for row in noisy_lidar_pose:
    #         f.write(' '.join(map(str, row)) + '\n')

    # print("Data has been appended to noisy_lidar_pose_beforeAlign.txt file.")
  
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl  
    N = noisy_lidar_pose.shape[0] 
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)
   
    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].
  
    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
  
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
   
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
 
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """
 

    Number of Agent:  2
    Number of Box: [18, 20]  
    box_idx_to_agent [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    coordinate of Agent: (2, 6)--->(2,4,4)
    pred_center_world_cat :(38, 3)

    """    
    
    # print("Number of Agent ",noisy_lidar_pose.shape[0])
    
    # print("Number of Box",pred_len)
    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 


    print("box_idx_to_agent",box_idx_to_agent)
   

    # print("coordinate of Agent",lidar_pose_noisy_tfm.shape)  
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    # print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag 
    thres_score = 0.5 


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

        
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)

  
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]
 
    neighbor_radius = 10.0  
    pre_neighbor_set = []
    candidate_radius = 2 
    # candidate_radius = 1  
    pre_candidate_set = []

    for box_id, pre_box_center in enumerate(pred_center_world_cat):
    #     box_neighbor_set = []
    #     # print(pred_center_allpair_dist[box_id])
    #     within_thres_neighbor_idx = (pred_center_allpair_dist[box_id] < neighbor_radius).nonzero()[0].tolist()
    #     # print(within_thres_neighbor_idx)
    #     for pre_nei in within_thres_neighbor_idx:
    #         if (box_idx_to_agent[pre_nei] == box_idx_to_agent[box_id] and pre_nei!=box_id):
    #             box_neighbor_set.append(pre_nei)
    #     pre_neighbor_set.append(box_neighbor_set)


        box_candidate_set = []   
        within_thres_candidate_idx = (pred_center_allpair_dist[box_id] < candidate_radius).nonzero()[0].tolist()
        for pre_can in within_thres_candidate_idx:
            if (box_idx_to_agent[pre_can] != box_idx_to_agent[box_id] and pre_can!=box_id):
                box_candidate_set.append(pre_can)
        pre_candidate_set.append(box_candidate_set)

    # print("pre_neighbor_set ", len(pre_neighbor_set))
    # print(pre_neighbor_set)

    print("pre_candidate_set ", len(pre_candidate_set), pred_len[0])
    print(pre_candidate_set)

    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0  
    for i in range(N):  
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i] 
    

    cluster_id = N # let the vertex id of object start from N 
    cluster_dict = OrderedDict()  
    remain_box = set(range(cum))  

    candidate_set = dict() 
    candidate_set_score = dict()
    # idx_set = set()
    # candidate_idx_set = set()

    target_graph_ids = list(range(pred_len[0]))  
    begin_id = 0
    end_id = pred_len[0]
    for i in range(1, N): 
        begin_id = end_id
        end_id += pred_len[i]
        idx_set = set()
        candidate_idx_set = set()
        print("target_graph_ids",target_graph_ids)
   
        print("box id range",begin_id, end_id)
        for box_idx in range(begin_id, end_id): 

            within_thres_idx_tensor_ws = (pred_center_allpair_dist[box_idx] < candidate_radius).nonzero()[0]
            within_thres_idx_list_ws = within_thres_idx_tensor_ws.tolist()
            # print(box_idx, within_thres_idx_list_ws)
            within_thres_idx_list_score = []  
            within_thres_idx_list_ws_in_target = []
            if len(within_thres_idx_list_ws) == 0:  # if it's a single box
                continue
                
            score_num = 0

            # if len(within_thres_idx_list) > 1:
            for candidate_idx in within_thres_idx_list_ws:  
                
                if candidate_idx not in target_graph_ids:  
                    continue
                score = compute_graph_similarity(box_idx, candidate_idx, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat)
                # score = compute_graph_similarity_only_distance(box_idx, candidate_idx, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat)
                # score = compute_graph_similarity_only_edge(box_idx, candidate_idx, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat)
                # print(box_idx, candidate_idx, score)
                # if score<thres_score:
                #     continue
                within_thres_idx_list_score.append(score)
                within_thres_idx_list_ws_in_target.append(candidate_idx)
                candidate_idx_set.add(candidate_idx)
                score_num += 1
            
            if score_num == 0:
                target_graph_ids.append(box_idx)

            idx_set.add(box_idx)
            candidate_set[box_idx] = within_thres_idx_list_ws_in_target
            candidate_set_score[box_idx] = within_thres_idx_list_score
            
        HK_result, unmatched_nodes = weighted_bipartite_matching_hopcroft_karp(idx_set, candidate_idx_set, candidate_set, candidate_set_score)

        if len(unmatched_nodes) > 0: 
            # print("unmatched_nodes:: ",unmatched_nodes)
            for un_node in unmatched_nodes:
                if un_node not in target_graph_ids:
                    target_graph_ids.append(un_node)
            # target_graph_ids += list(unmatched_nodes)

        # multi_weighted_bipartite_matching_hopcroft_karp(idx_set, candidate_idx_set, candidate_set, candidate_set_score, box_idx_to_agent)
        print(HK_result)
        
        for item in HK_result:
            if item[0] not in remain_box and item[1] not in remain_box: 
                continue
            if item[0] not in remain_box or item[1] not in remain_box: 
                newidx = item[0] if item[0] in remain_box else item[1]  
                oldidx = item[0] if item[0] not in remain_box else item[1] 
                for i in range(N, cluster_id):  # 
                    if oldidx in cluster_dict[i]['box_idx']:   # 
                        cluster_dict[i]['box_idx'].append(newidx)
                        cluster_dict[i]['box_center_world'].append(pred_center_world_cat[newidx])
                        cluster_dict[i]['box_yaw'].append(pred_yaw_world_cat[newidx])
                        cluster_dict[i]['box_yaw_varies'] = True if np.var(cluster_dict[i]['box_yaw']) > yaw_var_thres else False
                        cluster_dict[i]['active'] = True

                        if landmark_SE2:  #true
                            landmark = copy.deepcopy(pred_center_world_cat[box_idx]) 
                            landmark[2] = pred_yaw_world_cat[box_idx]
                        else:
                            landmark = pred_center_world_cat[box_idx][:2]
                        cluster_dict[i]['landmark'] = landmark  # [x, y, yaw] or [x, y]
                        cluster_dict[i]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

                        remain_box.remove(newidx)


                continue
       
            cluster_dict[cluster_id] = OrderedDict()
            cluster_dict[cluster_id]['box_idx'] = [item[0], item[1]]
            cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[item[0]], pred_center_world_cat[item[1]]]
            cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[item[0]], pred_yaw_world_cat[item[1]]]

            yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) #
            cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   # 
            cluster_dict[cluster_id]['active'] = True

            ########### adaptive_landmark ##################
            if landmark_SE2:  #true
                if adaptive_landmark and yaw_var > yaw_var_thres:
                    landmark = pred_center_world_cat[box_idx][:2]
                    for _box_idx in [item[0], item[1]]:
                        pred_certainty_cat[_box_idx] *= 2
                else:
                    landmark = copy.deepcopy(pred_center_world_cat[box_idx])  
                    landmark[2] = pred_yaw_world_cat[box_idx] 
            else:
                landmark = pred_center_world_cat[box_idx][:2]
            ##################################################

            cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] or [x, y]
            cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

            cluster_id += 1
            remain_box.remove(item[0])
            remain_box.remove(item[1])

    # print(cluster_dict)
   
    for i in range(N, cluster_id):
        print("box_idx",cluster_dict[i]['box_idx'])
    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N


    ########### abandon_hard_cases ##########
    """
        We should think what is hard cases for agent-object pose graph optimization
            1. Overlapping boxes are rare (landmark_num <= 3)
            2. Yaw angles differ a lot
    """
    # abandon_hard_cases = true
    if abandon_hard_cases:
        # case1: object num is smaller than 3，
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:  
                cluster_dict[landmark_id]['active'] = False

    """
     
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D()  #

    # Add agent to vertexs 
    for agent_id in range(agent_num):
        v_id = agent_id  
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]] 
        pose_np[2] = np.deg2rad(pose_np[2])  
        v_pose = g2o.SE2(pose_np) 
       
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

   
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,) 
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #
        #
        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
        #
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set
    for landmark_id in range(agent_num, vertex_num): 
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  

        if not cluster_dict[landmark_id]['active']:
            continue
      
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) 
                info = np.identity(3, dtype=np.float64) 
                if uncertainty_list is not None: 
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] 

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue
          
            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)  

    pose_new_list = []  
    for agent_id in range(agent_num): 
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) 

    refined_pose = np.array(pose_new_list)  
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source 

    # with open('/home/hz/code/opencood/logs/Other_results/noisy_lidar_pose_AfterAlign.txt', 'a') as f:
    #     for row in refined_pose:
    #         f.write(' '.join(map(str, row)) + '\n')
    # print("Data has been appended to noisy_lidar_pose_AfterAlign.txt file.")
    return refined_pose  


def box_alignment_relative_sample_np_direct(
            pred_corners_list,     
            noisy_lidar_pose,     
            uncertainty_list=None, 
            landmark_SE2=True,   
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,  
            drop_unsure_edge = False, 
            use_uncertainty = True,
            thres = 1.5,   
            yaw_var_thres = 0.2,   
            max_iterations = 1000):  
    """ Perform box alignment for one sample. 
    Correcting the relative pose.

    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        clean_lidar_poses:
            [N_cav1, 6], in degree
        
        noisy_lidar_poses:
            [N_cav1, 6], in degree

        uncertainty_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

        landmark_SE2:
            if True, the landmark is SE(2), otherwise R^2
        
        adaptive_landmark: (when landmark_SE2 = True)
            if True, landmark will turn to R^2 if yaw angles differ a lot

        normalize_uncertainty: bool
            if True, normalize the uncertainty
        
        abandon_hard_cases: bool
            if True, algorithm will just return original poses for hard cases

        drop_unsure_edge: bool

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """

    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl 
    N = noisy_lidar_pose.shape[0]
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)

    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].

    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]

    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian

    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian

    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """

    例如：
    Number of Agent:  2
    Number of Box: [18, 20] 
    box_idx_to_agent [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    coordinate of Agent: (2, 6)--->(2,4,4)
    pred_center_world_cat :(38, 3)

    """    
    
    # print("Number of Agent ",noisy_lidar_pose.shape[0])
    
    # print("Number of Box",pred_len)
    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 


    print("box_idx_to_agent",box_idx_to_agent)
   

    # print("coordinate of Agent",lidar_pose_noisy_tfm.shape) 
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    # print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag  


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

     
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)


    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]

    neighbor_radius = 10.0 
    pre_neighbor_set = []
    candidate_radius = l_a  
    pre_candidate_set = []

    for box_id, pre_box_center in enumerate(pred_center_world_cat):
        box_candidate_set = []   
        within_thres_candidate_idx = (pred_center_allpair_dist[box_id] < candidate_radius).nonzero()[0].tolist()
        for pre_can in within_thres_candidate_idx:
            if (box_idx_to_agent[pre_can] != box_idx_to_agent[box_id] and pre_can!=box_id):
                box_candidate_set.append(pre_can)
        pre_candidate_set.append(box_candidate_set)

    # print("pre_neighbor_set ", len(pre_neighbor_set))
    # print(pre_neighbor_set)

    print("pre_candidate_set ", len(pre_candidate_set), pred_len[0])
    print(pre_candidate_set)

    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0  
    for i in range(N):  
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]  
    

    cluster_id = N # let the vertex id of object start from N 
    cluster_dict = OrderedDict()  
    remain_box = set(range(cum))  

    candidate_set = dict() 
    candidate_set_score = dict()
    idx_set = set()
    candidate_idx_set = set()

    # target_graph_ids = list(range(pred_len[0]))  
    

    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned  
            continue

        within_thres_idx_tensor_ws = (pred_center_allpair_dist[box_idx] < candidate_radius).nonzero()[0]
        within_thres_idx_list_ws = within_thres_idx_tensor_ws.tolist()  # 
        # print(box_idx, within_thres_idx_list_ws)
        within_thres_idx_list_score = []  # 
        if len(within_thres_idx_list_ws) == 0:  # if it's a single box
            continue
        
        explored = [box_idx]
      
        agent_candidates = [] #
        agent_candidates_score = [] # 
        for i in range(N-1):
            agent_candidates.append([])
            agent_candidates_score.append([])
        for candidate_idx in within_thres_idx_list_ws:  #
            if candidate_idx not in remain_box:  # 
                continue
            score = compute_graph_similarity(box_idx, candidate_idx, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat)
            
            agent_candidates[box_idx_to_agent[candidate_idx]-1].append(candidate_idx)
            agent_candidates_score[box_idx_to_agent[candidate_idx]-1].append(score)
        

        for i in range(N-1):
            if len(agent_candidates[i]) == 0:
                continue
 
            max_idx = agent_candidates_score[i].index(max(agent_candidates_score[i]))
            if(max(agent_candidates_score[i]) > 1e-3):
                continue
            explored.append(agent_candidates[i][max_idx])
    
       
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue

        cluster_box_idxs = explored  

        cluster_dict[cluster_id] = OrderedDict()  
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs] 
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # 存坐标 coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs] 

        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) 
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   
        cluster_dict[cluster_id]['active'] = True 


        ########### adaptive_landmark ##################
        if landmark_SE2:  #true
            if adaptive_landmark and yaw_var > yaw_var_thres:
                landmark = pred_center_world_cat[box_idx][:2]
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])  
                landmark[2] = pred_yaw_world_cat[box_idx]  
        else:
            landmark = pred_center_world_cat[box_idx][:2]
        ##################################################


        cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] or [x, y]
        cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

        DEBUG = False
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(cluster_dict[cluster_id]['box_center_world'])
            ic(cluster_dict[cluster_id]['box_yaw'])
            ic(cluster_dict[cluster_id]['landmark'])
        

        cluster_id += 1
    
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    print(cluster_dict)
    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N


    ########### abandon_hard_cases ##########
    """
        We should think what is hard cases for agent-object pose graph optimization
            1. Overlapping boxes are rare (landmark_num <= 3)
            2. Yaw angles differ a lot
    """
    # abandon_hard_cases = true
    if abandon_hard_cases:
        # case1: object num is smaller than 3，
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:  
                cluster_dict[landmark_id]['active'] = False

    """

        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D() 

    # Add agent to vertexs 
    for agent_id in range(agent_num):
        v_id = agent_id  
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]] 
        pose_np[2] = np.deg2rad(pose_np[2])  # radians 
        v_pose = g2o.SE2(pose_np)  
    
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

   
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,)
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  

        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
       
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set 
    for landmark_id in range(agent_num, vertex_num): 
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2'] 

        if not cluster_dict[landmark_id]['active']:
            continue
   
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) 
                info = np.identity(3, dtype=np.float64) 
                if uncertainty_list is not None: 
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] 

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue

            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations) 

    pose_new_list = []  
    for agent_id in range(agent_num): 
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) 

    refined_pose = np.array(pose_new_list)  
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source 

    return refined_pose  



def box_alignment_relative_sample_np_old(   
            pred_corners_list,    
            noisy_lidar_pose,     
            uncertainty_list=None, 
            landmark_SE2=True,  
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,  
            drop_unsure_edge = False, 
            use_uncertainty = True,
            thres = 1.5,     
            yaw_var_thres = 0.2,   
            max_iterations = 1000): 
    """ Perform box alignment for one sample. 
    Correcting the relative pose.

    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        clean_lidar_poses:
            [N_cav1, 6], in degree
        
        noisy_lidar_poses:
            [N_cav1, 6], in degree

        uncertainty_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

        landmark_SE2:
            if True, the landmark is SE(2), otherwise R^2
        
        adaptive_landmark: (when landmark_SE2 = True)
            if True, landmark will turn to R^2 if yaw angles differ a lot

        normalize_uncertainty: bool
            if True, normalize the uncertainty
        
        abandon_hard_cases: bool
            if True, algorithm will just return original poses for hard cases

        drop_unsure_edge: bool

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """
 
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl  
    N = noisy_lidar_pose.shape[0]
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)

    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].

    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
  
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
  
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
   
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """

    Number of Agent:  2
    Number of Box: [18, 20]
    box_idx_to_agent [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    coordinate of Agent: (2, 6)--->(2,4,4)
    pred_center_world_cat :(38, 3)

    """    
    
    print("Number of Agent ",noisy_lidar_pose.shape[0])
    
    print("Number of Box",pred_len)
    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 


    print("box_idx_to_agent",box_idx_to_agent)


    print("coordinate of Agent",lidar_pose_noisy_tfm.shape)  
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag 


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

   
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)


    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]

    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0  
    for i in range(N): 
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]  
    print(pred_center_allpair_dist)
    print(N)

    cluster_id = N # let the vertex id of object start from N 
    cluster_dict = OrderedDict() 
    remain_box = set(range(cum)) 
    #遍历每一个bounding box,
    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned 
            continue
        
 
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
        within_thres_idx_list = within_thres_idx_tensor.tolist()
        
      
        if len(within_thres_idx_list) == 0:  # if it's a single box
            continue
        
        #从 within_thres_idx_list 
        # start from within_thres_idx_list, find new box added to the cluster
        explored = [box_idx]
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]  

        while unexplored:
            idx = unexplored[0]
            within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0] 
            within_thres_idx_list = within_thres_idx_tensor.tolist()
           
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            unexplored.remove(idx)
            explored.append(idx)
        
       
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue
        
        cluster_box_idxs = explored  

        cluster_dict[cluster_id] = OrderedDict()  
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]  
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs] 
        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) 
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   
        cluster_dict[cluster_id]['active'] = True 


        ########### adaptive_landmark ##################
        if landmark_SE2:  #true
            if adaptive_landmark and yaw_var > yaw_var_thres:
                landmark = pred_center_world_cat[box_idx][:2]
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])  
                landmark[2] = pred_yaw_world_cat[box_idx] 
        else:
            landmark = pred_center_world_cat[box_idx][:2]
        ##################################################


        cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] or [x, y]
        cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

        DEBUG = False
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(cluster_dict[cluster_id]['box_center_world'])
            ic(cluster_dict[cluster_id]['box_yaw'])
            ic(cluster_dict[cluster_id]['landmark'])
        

        cluster_id += 1
    
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    
    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N


    ########### abandon_hard_cases ##########
    """
        We should think what is hard cases for agent-object pose graph optimization
            1. Overlapping boxes are rare (landmark_num <= 3)
            2. Yaw angles differ a lot
    """
    # abandon_hard_cases = true
    if abandon_hard_cases:
        # case1: object num is smaller than 3，
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']: 
                cluster_dict[landmark_id]['active'] = False

    """

        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D() 

    # Add agent to vertexs 
    for agent_id in range(agent_num):
        v_id = agent_id 
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]]  
        pose_np[2] = np.deg2rad(pose_np[2])  # radians 
        v_pose = g2o.SE2(pose_np)  
     

        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add object to vertexs 
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,) 
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #
        #
        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
        #
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set 
    for landmark_id in range(agent_num, vertex_num):  #
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #
    
        if not cluster_dict[landmark_id]['active']:
            continue

        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  #
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) #
                info = np.identity(3, dtype=np.float64)  #
                if uncertainty_list is not None:  #
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] #

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  #
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue

            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)  #

    pose_new_list = []  #
    for agent_id in range(agent_num):  #
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) 

    refined_pose = np.array(pose_new_list)  #
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source 

    return refined_pose  #

def box_alignment_relative_np(pred_corner3d_list, 
                              uncertainty_list, 
                              lidar_poses, 
                              record_len, 
                              **kwargs):
    """
    Args:
        pred_corner3d_list: list of tensors, with shape [[N1_object, 8, 3], [N2_object, 8, 3], ...,[N_sumcav_object, 8, 3]]
            box in each agent's coordinate. (proj_first=False)
        
        pred_box3d_list: not necessary
            list of tensors, with shape [[N1_object, 7], [N2_object, 7], ...,[N_sumcav_object, 7]]

        scores_list: list of tensor, [[N1_object,], [N2_object,], ...,[N_sumcav_object,]]
            box confidence score.

        lidar_poses: torch.Tensor [sum(cav), 6]

        record_len: torch.Tensor
    Returns:
        refined_lidar_pose: torch.Tensor [sum(cav), 6]
    """
    refined_lidar_pose = []
    start_idx = 0
    for b in record_len:
        refined_lidar_pose.append(
            box_alignment_relative_sample_np(
                pred_corner3d_list[start_idx: start_idx + b],
                lidar_poses[start_idx: start_idx + b],
                uncertainty_list= None if uncertainty_list is None else uncertainty_list[start_idx: start_idx + b],
                **kwargs
            )
        )
        start_idx += b

    return np.cat(refined_lidar_pose, axis=0)


