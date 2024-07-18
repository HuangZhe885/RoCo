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

# 计算从a1到a2的相对位姿 和 从b1到b2的相对位姿，然后计算两者的差别
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
    agent_idx = box_idx_to_agent[idx]  #获取idx对应的代理id
    agent_candidate = box_idx_to_agent[idx_candidate]  #获取idx_candidate对应的代理id

    begin_idx = sum(pred_len[:agent_idx])
    end_idx = begin_idx + pred_len[agent_idx]
    graph_idx = []
    graph_candidate = []
    for i in range(begin_idx, end_idx):
        if box_idx_to_agent[i] == agent_idx and idx != i:  # 同属于一个agent
            if len(pre_candidate_set[i]) >= 1:  # 该邻居节点的候选集不为空
                count = 0
                for j in pre_candidate_set[i]:
                    if box_idx_to_agent[j] == agent_candidate:
                        count += 1
                    
                if count == 1:  # 只有一个属于agent_candidate
                    graph_idx.append(i)
                    graph_candidate.append(j)

    score = 0
    for i in range(len(graph_idx)):
        edge_diff = compute_edge(idx, graph_idx[i], idx_candidate, graph_candidate[i], pred_center_world_cat) # / len(graph_candidate[i])
        score += math.exp(-np.linalg.norm(edge_diff))
    # 求平均
    if len(graph_idx) > 0:
        score /= len(graph_idx)
        # print("graph score:", score)

    return score

def compute_graph_similarity_only_distance(idx, idx_candidate, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat):

    distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
    score = distance_score
    return score

# 计算idx为中心的子图与以idx_candidate为中心的子图之间的相似度
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
    agent_idx = box_idx_to_agent[idx]  #获取idx对应的代理id
    agent_candidate = box_idx_to_agent[idx_candidate]  #获取idx_candidate对应的代理id

    # 选取idx为中心的子图的其他顶点，选择的标准是同属于一个agent，且对应候选集中的box中属于该agent的只有一个
    begin_idx = sum(pred_len[:agent_idx])
    end_idx = begin_idx + pred_len[agent_idx]
    graph_idx = []
    graph_candidate = []
    for i in range(begin_idx, end_idx):
        if box_idx_to_agent[i] == agent_idx and idx != i:  # 同属于一个agent
            if len(pre_candidate_set[i]) >= 1:  # 该邻居节点的候选集不为空
                # 判断该邻居节点的候选集中是否只有一个属于该agent，如果没有该agent或有多个属于该agent，则跳过
                count = 0
                for j in pre_candidate_set[i]:
                    if box_idx_to_agent[j] == agent_candidate:
                        count += 1
                    
                if count == 1:  # 只有一个属于agent_candidate
                    graph_idx.append(i)
                    graph_candidate.append(j)
    
    if (len(graph_idx) == 0):  # 表示所有的agent暂时都没有确定的邻居节点
        print("--------------no right graph----------------")
        print("--------------relax the thera----------------")
        # 当前agent的检测框数量要大于1，否则直接返回-1

        # 下面是松弛条件
        # if pred_len[agent_idx] <= 1:
        #     return -1
        # #
        # limits = 1
        # while(len(graph_idx) <= 1):
        #     limits += 1
        #     for i in range(begin_idx, end_idx):
        #         if box_idx_to_agent[i] == agent_idx and idx != i:  # 同属于一个agent
        #             if len(pre_candidate_set[i]) >= 1 and i not in graph_idx:  # 该邻居节点的候选集不为空,且不在图中
        #                 count = 0
        #                 for j in pre_candidate_set[i]:
        #                     if box_idx_to_agent[j] == agent_candidate:
        #                         count += 1
        #                 if count == limits:  # 只有一个属于agent_candidate
        #                     graph_idx.append(i)
        #                     graph_candidate.append(j)
        #     if (limits > len(pred_len)): # 表示没有合适的
        #         return -1


    score = 0
    for i in range(len(graph_idx)):
        edge_diff = compute_edge(idx, graph_idx[i], idx_candidate, graph_candidate[i], pred_center_world_cat) # / len(graph_candidate[i])
        score += math.exp(-np.linalg.norm(edge_diff))
    # 求平均
    if len(graph_idx) > 0:
        score /= len(graph_idx)
        print("graph score:", score)

    if len(graph_idx) == 0:
        distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
        score = distance_score
        return score
    
    
    distance_score = math.exp(-np.linalg.norm(pred_center_world_cat[idx] - pred_center_world_cat[idx_candidate]))
    score += distance_score
    # 如果只用edge，就注释上面一行，但效果会很差
    
    # print(idx, idx_candidate, score)
    
    return score

    
    
def weighted_bipartite_matching_hopcroft_karp(idx_set: set, candidate_idx_set: set, candidate_set: dict, candidate_set_score: dict):

    print("idx_set", idx_set)
    print("candidate_idx_set", candidate_idx_set)
    print("candidate_set", candidate_set)
    print("candidate_set_score", candidate_set_score)
    # 创建一个带权重的二部图
    G = nx.Graph()


    # 添加代理节点
    left_nodes = list(idx_set)
    right_nodes = list(candidate_idx_set)
    # print("left_nodes",left_nodes)
    # print("right_nodes",right_nodes)

    # 添加节点到图中
    G.add_nodes_from(left_nodes, bipartite=0)  # 代理节点
    G.add_nodes_from(right_nodes, bipartite=1)  # 目标节点

    # print("nodes ::", G.nodes())
    # print(candidate_set)
    # print(candidate_set_score)

    # 添加权重边
    for i in range(len(left_nodes)):
        for j in range(len(candidate_set[left_nodes[i]])):
            G.add_edge(left_nodes[i], candidate_set[left_nodes[i]][j], weight=candidate_set_score[left_nodes[i]][j])

    # 添加边并赋予权重
    # for i in range(left_nodes):
    #     for j in range(num_targets):
    #         G.add_edge(i, j + num_agents, weight=weights[i, j])  # 注意将权重取负号，因为 NetworkX 默认找最小权重匹配

    # 使用 Hopcroft-Karp 算法求解
    matching = nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes=left_nodes)

    # 提取匹配结果
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

    # 3个及以上个agent的情况，需要图融合
    

    return 

# 使用HK来做最终的匹配，bug
def box_alignment_relative_sample_np(
            pred_corners_list,     #当前代理的3D框坐标列表
            noisy_lidar_pose,      #代理的姿态信息，包括位置和朝向
            uncertainty_list=None, 
            landmark_SE2=True,   #landmark类型为SE2
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,  ##放弃处理困难的框
            drop_unsure_edge = False, #放弃处理不确定的情况
            use_uncertainty = True,
            thres = 1.5,     #确定相邻框的距离阈值
            yaw_var_thres = 0.2,    #用于确定框姿态角差异的阈值
            max_iterations = 1000):  #最优迭代的最大次数
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
    
    # 将数据附加到TXT文件的末尾
    # with open('/home/hz/code/opencood/logs/Other_results/noisy_lidar_pose_beforeAlign.txt', 'a') as f:
    #     for row in noisy_lidar_pose:
    #         f.write(' '.join(map(str, row)) + '\n')

    # print("Data has been appended to noisy_lidar_pose_beforeAlign.txt file.")
    # print("图优化开始=====")
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl  定义框的尺寸顺序为长度，宽度，高度
    N = noisy_lidar_pose.shape[0] #代理的数量
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)
    #创建列表，表示是否检测到Object
    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].
    #转换为世界坐标系
    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
    #计算每个box的中心点坐标
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box的中心点在世界坐标系中的坐标
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box中心点坐标的均值
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """
    传进来一共两个agent
    box_idx_to_agent 用于计算每个框box与代理agent之间建立映射关系,内容为每个框对应代理的索引
    例如：
    Number of Agent:  2
    Number of Box: [18, 20]  属于索引为0的代理的box一共有18个,属于第二个代理的box一共有20个 
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
    #分别将中心点坐标，世界坐标系的中心点坐标，。。角度整个到单个数组中

    # print("coordinate of Agent",lidar_pose_noisy_tfm.shape)  #查看Agent的位姿
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    # print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag  对角线的平方
    thres_score = 0.5  #阈值分数


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  #代理不确定性对数
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

        #将对数不确定性转化为确定性，然后归一化
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)

    # 计算所有box中心点之间的距离矩阵
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]
    # 构建每个box的邻居集合，每个agent的每个box周围5m范围内的box为邻居，
    neighbor_radius = 10.0  # 邻居构建的阈值
    pre_neighbor_set = []
    candidate_radius = 2  # 候选集构建的阈值
    # candidate_radius = 1  # 候选集构建的阈值
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
    cum = 0  #用于在循环中记录累计box的数量
    for i in range(N):  #遍历所有的代理N  ，将距离矩阵两个对角线子矩阵都变成最大值，保证每个代理的box之间的距离很大，不会被分到同一个聚类中
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]  #更新b当前bounding box的数量
    

    cluster_id = N # let the vertex id of object start from N 聚类起始的值为N
    cluster_dict = OrderedDict()  #创建有序字典，保存聚类信息
    remain_box = set(range(cum))  #创建集合，包含所有bounding box的索引，表示没有被分到任何聚类中。

    candidate_set = dict() 
    candidate_set_score = dict()
    # idx_set = set()
    # candidate_idx_set = set()

    target_graph_ids = list(range(pred_len[0]))  # 初始时设定0号agent图，每次完成匹配后将大图融合进入target_graph_ids
    begin_id = 0
    end_id = pred_len[0]
    for i in range(1, N):  # 将后续agent的图融合进入target_graph_ids
        begin_id = end_id
        end_id += pred_len[i]
        idx_set = set()
        candidate_idx_set = set()
        print("target_graph_ids",target_graph_ids)
        #遍历每一个bounding box,
        print("box id range",begin_id, end_id)
        for box_idx in range(begin_id, end_id): 

            within_thres_idx_tensor_ws = (pred_center_allpair_dist[box_idx] < candidate_radius).nonzero()[0]
            within_thres_idx_list_ws = within_thres_idx_tensor_ws.tolist()
            # print(box_idx, within_thres_idx_list_ws)
            within_thres_idx_list_score = []  # 记录候选集中每个box的得分
            within_thres_idx_list_ws_in_target = []
            if len(within_thres_idx_list_ws) == 0:  # if it's a single box
                continue
                
            score_num = 0

            # if len(within_thres_idx_list) > 1:
            for candidate_idx in within_thres_idx_list_ws:  # 对候选集中的每个box计算得分
                
                if candidate_idx not in target_graph_ids:  # 如果候选集中的box不在target_graph_ids中，跳过
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

        if len(unmatched_nodes) > 0: # 存在未匹配的节点、
            # print("unmatched_nodes:: ",unmatched_nodes)
            for un_node in unmatched_nodes:
                if un_node not in target_graph_ids:
                    target_graph_ids.append(un_node)
            # target_graph_ids += list(unmatched_nodes)

        # multi_weighted_bipartite_matching_hopcroft_karp(idx_set, candidate_idx_set, candidate_set, candidate_set_score, box_idx_to_agent)
        print(HK_result)
        
        for item in HK_result:
            if item[0] not in remain_box and item[1] not in remain_box: # 当匹配的两个结果都已处理过
                continue
            if item[0] not in remain_box or item[1] not in remain_box: # 表示其中一个已经被处理过，但需要将新匹配放入cluster_dict中
                newidx = item[0] if item[0] in remain_box else item[1]  # 新放入的id
                oldidx = item[0] if item[0] not in remain_box else item[1] # 已存在的id
                for i in range(N, cluster_id):  # 遍历所有的聚类
                    if oldidx in cluster_dict[i]['box_idx']:   # 找到已存在的id所在的聚类，将新放入的id放入该聚类中
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
            # 如果两个都没有被处理，那么创建一个匹配结果
            cluster_dict[cluster_id] = OrderedDict()
            cluster_dict[cluster_id]['box_idx'] = [item[0], item[1]]
            cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[item[0]], pred_center_world_cat[item[1]]]
            cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[item[0]], pred_yaw_world_cat[item[1]]]

            yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) #计算角度方差
            cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   # 方差是否大于阈值，如果是，则标记角度变化大
            cluster_dict[cluster_id]['active'] = True #当前聚类为有效

            ########### adaptive_landmark ##################
            if landmark_SE2:  #true
                if adaptive_landmark and yaw_var > yaw_var_thres:
                    landmark = pred_center_world_cat[box_idx][:2]
                    for _box_idx in [item[0], item[1]]:
                        pred_certainty_cat[_box_idx] *= 2
                else:
                    landmark = copy.deepcopy(pred_center_world_cat[box_idx])  #将landmark设置为当前bounding box的中心坐标和角度
                    landmark[2] = pred_yaw_world_cat[box_idx]  #更新landmark的角度信息，等于当前bounding box的角度
            else:
                landmark = pred_center_world_cat[box_idx][:2]
            ##################################################

            cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] or [x, y]
            cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

            cluster_id += 1
            remain_box.remove(item[0])
            remain_box.remove(item[1])

    # print(cluster_dict)
    # 打印cluster_dict中的每一个'box_idx'
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
        # case1: object num is smaller than 3，如果聚类物体数量小于3，说明物体较少，难以优化位姿，直接返回原始代理的pose信息
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，如果超过一半的物体角度差异较大，直接返回代理的pose信息，跳出后续的图优化
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:  #如果某物体的角度差异较大，将该物体标记为不活跃，不参与图优化
                cluster_dict[landmark_id]['active'] = False

    """
        开始图优化
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D()  #创建一个二维姿态图用于姿态优化

    # Add agent to vertexs 将代理姿态添加到姿态图中
    for agent_id in range(agent_num):
        v_id = agent_id  #为每个代理分配唯一的id
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]]  #获取代理的位置和朝向
        pose_np[2] = np.deg2rad(pose_np[2])  # radians 将朝向信息从度数转为弧度
        v_pose = g2o.SE2(pose_np)   #创建一个SE2姿态对象
        #如果是第一个代理，将代理的姿态信息添加到图中，并将其标记为固定，它的位姿信息不会变
        # 将其他代理的姿态信息添加到图中，并将其标记为不固定，可以在后续的优化中进行调整
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add object to vertexs 将物体（landmark）的姿态添加到姿态图中
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,) 获取物体的位置朝向
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #判断物体的地标类型是否为SE2
        #如果是SE2，，创建SE2对象，并使用地标信息初始化
        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
        #将物体的姿态图信息添加到图中，不标记为固定，类型为SE2
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set 将代理与物体之间的关联关系添加到边集
    for landmark_id in range(agent_num, vertex_num):  #遍历所有物体
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #获取物体的地标类型
        #如果物体地标类型不是活跃的，跳过，不与之关联
        if not cluster_dict[landmark_id]['active']:
            continue
        #遍历所有与物体关联的框
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  #找到与bounding box关联的代理id
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) #创建一个SE2类型的边，使用框的位置和朝向信息初始化
                info = np.identity(3, dtype=np.float64)  #创建一个3*3的单位矩阵作为信息矩阵，用于优化权重
                if uncertainty_list is not None:  #如果提供了不确定性信息
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] #根据不确定信息更新信息矩阵的对角线元素

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  #如果不确定性总和小于100，跳过这个边
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue
            #将边信息添加到姿态图中，表示代理和物体之间的关联
            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)  #对姿态图进行优化，最大化迭代次数

    pose_new_list = []  #创建一个空列表，用于存储优化后的姿态信息
    for agent_id in range(agent_num):  #遍历所有代理
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) #获取并添加每个代理优化后的姿态信息

    refined_pose = np.array(pose_new_list)  #将优化后的姿态信息转换为Numpy数组
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source #将姿态信息的朝向从弧度转为度数
    # 将数据附加到TXT文件的末尾
    # with open('/home/hz/code/opencood/logs/Other_results/noisy_lidar_pose_AfterAlign.txt', 'a') as f:
    #     for row in refined_pose:
    #         f.write(' '.join(map(str, row)) + '\n')
    # print("Data has been appended to noisy_lidar_pose_AfterAlign.txt file.")
    return refined_pose  #返回优化后的姿态信息，包括位置和朝向

# 直接使用图结构来匹配和过滤
def box_alignment_relative_sample_np_direct(
            pred_corners_list,     #当前代理的3D框坐标列表
            noisy_lidar_pose,      #代理的姿态信息，包括位置和朝向
            uncertainty_list=None, 
            landmark_SE2=True,   #landmark类型为SE2
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,  ##放弃处理困难的框
            drop_unsure_edge = False, #放弃处理不确定的情况
            use_uncertainty = True,
            thres = 1.5,     #确定相邻框的距离阈值
            yaw_var_thres = 0.2,    #用于确定框姿态角差异的阈值
            max_iterations = 1000):  #最优迭代的最大次数
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
    # print("图优化开始=====")
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl  定义框的尺寸顺序为长度，宽度，高度
    N = noisy_lidar_pose.shape[0] #代理的数量
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)
    #创建列表，表示是否检测到Object
    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].
    #转换为世界坐标系
    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
    #计算每个box的中心点坐标
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box的中心点在世界坐标系中的坐标
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box中心点坐标的均值
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """
    传进来一共两个agent
    box_idx_to_agent 用于计算每个框box与代理agent之间建立映射关系,内容为每个框对应代理的索引
    例如：
    Number of Agent:  2
    Number of Box: [18, 20]  属于索引为0的代理的box一共有18个,属于第二个代理的box一共有20个 
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
    #分别将中心点坐标，世界坐标系的中心点坐标，。。角度整个到单个数组中

    # print("coordinate of Agent",lidar_pose_noisy_tfm.shape)  #查看Agent的位姿
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    # print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag  对角线的平方


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  #代理不确定性对数
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

        #将对数不确定性转化为确定性，然后归一化
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)

    # 计算所有box中心点之间的距离矩阵
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]
    # 构建每个box的邻居集合，每个agent的每个box周围5m范围内的box为邻居，
    neighbor_radius = 10.0  # 邻居构建的阈值
    pre_neighbor_set = []
    candidate_radius = l_a  # 候选集构建的阈值
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
    cum = 0  #用于在循环中记录累计box的数量
    for i in range(N):  #遍历所有的代理N  ，将距离矩阵两个对角线子矩阵都变成最大值，保证每个代理的box之间的距离很大，不会被分到同一个聚类中
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]  #更新b当前bounding box的数量
    

    cluster_id = N # let the vertex id of object start from N 聚类起始的值为N
    cluster_dict = OrderedDict()  #创建有序字典，保存聚类信息
    remain_box = set(range(cum))  #创建集合，包含所有bounding box的索引，表示没有被分到任何聚类中。

    candidate_set = dict() 
    candidate_set_score = dict()
    idx_set = set()
    candidate_idx_set = set()

    # target_graph_ids = list(range(pred_len[0]))  # 初始时设定0号agent图，每次完成匹配后将大图融合进入target_graph_ids
    
    #遍历每一个bounding box,
    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned  如果ID没有在 remain_box里，就跳过，表示已经被处理过
            continue

        within_thres_idx_tensor_ws = (pred_center_allpair_dist[box_idx] < candidate_radius).nonzero()[0]
        within_thres_idx_list_ws = within_thres_idx_tensor_ws.tolist()  # 候选集中的box的索引列表
        # print(box_idx, within_thres_idx_list_ws)
        within_thres_idx_list_score = []  # 记录候选集中每个box的得分
        if len(within_thres_idx_list_ws) == 0:  # if it's a single box
            continue
        
        explored = [box_idx]
      
        agent_candidates = [] # 记录当前box属于各个代理的候选集
        agent_candidates_score = [] # 记录当前box属于各个代理的候选集
        for i in range(N-1):
            agent_candidates.append([])
            agent_candidates_score.append([])
        for candidate_idx in within_thres_idx_list_ws:  # 对候选集中的每个box计算得分
            if candidate_idx not in remain_box:  # 如果候选集中的box不在remain_box中，跳过
                continue
            score = compute_graph_similarity(box_idx, candidate_idx, pre_candidate_set, pred_len, box_idx_to_agent, pred_center_world_cat)
            
            agent_candidates[box_idx_to_agent[candidate_idx]-1].append(candidate_idx)
            agent_candidates_score[box_idx_to_agent[candidate_idx]-1].append(score)
        

        for i in range(N-1):
            if len(agent_candidates[i]) == 0:
                continue
            # 获取agent_candidates[i]中的最大值对应的索引
            max_idx = agent_candidates_score[i].index(max(agent_candidates_score[i]))
            if(max(agent_candidates_score[i]) > 1e-3):
                continue
            explored.append(agent_candidates[i][max_idx])
    
        # 如果只有一个索引，表示这个bounding box没有相邻的框，所以被单独分成一个聚类
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue

        cluster_box_idxs = explored  #将探索完成的bounding box列表作为当前聚类的box索引列表

        cluster_dict[cluster_id] = OrderedDict()  #创建有序字典，存储聚类信息
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]  #存储box的索引列表
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # 存坐标 coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs] #存角度

        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) #计算角度方差
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   # 方差是否大于阈值，如果是，则标记角度变化大
        cluster_dict[cluster_id]['active'] = True #当前聚类为有效


        ########### adaptive_landmark ##################
        if landmark_SE2:  #true
            if adaptive_landmark and yaw_var > yaw_var_thres:
                landmark = pred_center_world_cat[box_idx][:2]
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])  #将landmark设置为当前bounding box的中心坐标和角度
                landmark[2] = pred_yaw_world_cat[box_idx]  #更新landmark的角度信息，等于当前bounding box的角度
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
        
        # 递增聚类的ID，准备处理下一个聚类
        cluster_id += 1
        #遍历bounding box索引列表，移除已经分配到聚类中的bounding box
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
        # case1: object num is smaller than 3，如果聚类物体数量小于3，说明物体较少，难以优化位姿，直接返回原始代理的pose信息
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，如果超过一半的物体角度差异较大，直接返回代理的pose信息，跳出后续的图优化
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:  #如果某物体的角度差异较大，将该物体标记为不活跃，不参与图优化
                cluster_dict[landmark_id]['active'] = False

    """
        开始图优化
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D()  #创建一个二维姿态图用于姿态优化

    # Add agent to vertexs 将代理姿态添加到姿态图中
    for agent_id in range(agent_num):
        v_id = agent_id  #为每个代理分配唯一的id
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]]  #获取代理的位置和朝向
        pose_np[2] = np.deg2rad(pose_np[2])  # radians 将朝向信息从度数转为弧度
        v_pose = g2o.SE2(pose_np)   #创建一个SE2姿态对象
        #如果是第一个代理，将代理的姿态信息添加到图中，并将其标记为固定，它的位姿信息不会变
        # 将其他代理的姿态信息添加到图中，并将其标记为不固定，可以在后续的优化中进行调整
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add object to vertexs 将物体（landmark）的姿态添加到姿态图中
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,) 获取物体的位置朝向
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #判断物体的地标类型是否为SE2
        #如果是SE2，，创建SE2对象，并使用地标信息初始化
        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
        #将物体的姿态图信息添加到图中，不标记为固定，类型为SE2
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set 将代理与物体之间的关联关系添加到边集
    for landmark_id in range(agent_num, vertex_num):  #遍历所有物体
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #获取物体的地标类型
        #如果物体地标类型不是活跃的，跳过，不与之关联
        if not cluster_dict[landmark_id]['active']:
            continue
        #遍历所有与物体关联的框
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  #找到与bounding box关联的代理id
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) #创建一个SE2类型的边，使用框的位置和朝向信息初始化
                info = np.identity(3, dtype=np.float64)  #创建一个3*3的单位矩阵作为信息矩阵，用于优化权重
                if uncertainty_list is not None:  #如果提供了不确定性信息
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] #根据不确定信息更新信息矩阵的对角线元素

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  #如果不确定性总和小于100，跳过这个边
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue
            #将边信息添加到姿态图中，表示代理和物体之间的关联
            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)  #对姿态图进行优化，最大化迭代次数

    pose_new_list = []  #创建一个空列表，用于存储优化后的姿态信息
    for agent_id in range(agent_num):  #遍历所有代理
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) #获取并添加每个代理优化后的姿态信息

    refined_pose = np.array(pose_new_list)  #将优化后的姿态信息转换为Numpy数组
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source #将姿态信息的朝向从弧度转为度数

    return refined_pose  #返回优化后的姿态信息，包括位置和朝向


# 原始代码中的对齐与图优化
def box_alignment_relative_sample_np_old(   # 原始代码中的对齐与图优化
            pred_corners_list,     #当前代理的3D框坐标列表
            noisy_lidar_pose,      #代理的姿态信息，包括位置和朝向
            uncertainty_list=None, 
            landmark_SE2=True,   #landmark类型为SE2
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,  ##放弃处理困难的框
            drop_unsure_edge = False, #放弃处理不确定的情况
            use_uncertainty = True,
            thres = 1.5,     #确定相邻框的距离阈值
            yaw_var_thres = 0.2,    #用于确定框姿态角差异的阈值
            max_iterations = 1000):  #最优迭代的最大次数
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
    print("图优化开始=====")
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl  定义框的尺寸顺序为长度，宽度，高度
    N = noisy_lidar_pose.shape[0] #代理的数量
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)
    #创建列表，表示是否检测到Object
    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].
    #转换为世界坐标系
    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
    #计算每个box的中心点坐标
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box的中心点在世界坐标系中的坐标
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
    #计算每个box中心点坐标的均值
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    """
    传进来一共两个代理
    box_idx_to_agent 用于计算每个框box与代理agent之间建立映射关系,内容为每个框对应代理的索引
    例如：
    Number of Agent:  2
    Number of Box: [18, 20]  属于索引为0的代理的box一共有18个,属于第二个代理的box一共有20个 
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
    #分别将中心点坐标，世界坐标系的中心点坐标，。。角度整个到单个数组中

    print("coordinate of Agent",lidar_pose_noisy_tfm.shape)  #查看Agent的位姿
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    print("pred_center_world_cat",pred_center_world_cat.shape)
    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag  对角线的平方


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)  #代理不确定性对数
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square    

        #将对数不确定性转化为确定性，然后归一化
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)

    # 计算所有box中心点之间的距离矩阵
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]

    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0  #用于在循环中记录累计box的数量
    for i in range(N):  #遍历所有的代理N  ，将距离矩阵两个对角线子矩阵都变成最大值，保证每个代理的box之间的距离很大，不会被分到同一个聚类中
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]  #更新b当前bounding box的数量
    print(pred_center_allpair_dist)
    print(N)

    cluster_id = N # let the vertex id of object start from N 聚类起始的值为N
    cluster_dict = OrderedDict()  #创建有序字典，保存聚类信息
    remain_box = set(range(cum))  #创建集合，包含所有bounding box的索引，表示没有被分到任何聚类中。
    #遍历每一个bounding box,
    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned  如果ID没有在 remain_box里，就跳过，表示已经被处理过
            continue
        
        #找到与当前bounding box距离在阈值内的其他box的索引，将他们存储
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
        within_thres_idx_list = within_thres_idx_tensor.tolist()
        
        #如果没有距离当前的bounding box的框，继续下一个box处理
        if len(within_thres_idx_list) == 0:  # if it's a single box
            continue
        
        #从 within_thres_idx_list 开始，找到所有距离在阈值内的box，并添加到同一个聚类中。
        # start from within_thres_idx_list, find new box added to the cluster
        explored = [box_idx]
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]  #存储尚未探索的相邻框的索引

        while unexplored:
            idx = unexplored[0]
            within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0] #重新计算
            within_thres_idx_list = within_thres_idx_tensor.tolist()
            #检查所有新找到的box，并添加到explored列表中，同时从unexplored中移除
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            unexplored.remove(idx)
            explored.append(idx)
        
        # 如果只有一个索引，表示这个bounding box没有相邻的框，所以被单独分成一个聚类
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue
        
        cluster_box_idxs = explored  #将探索完成的bounding box列表作为当前聚类的box索引列表

        cluster_dict[cluster_id] = OrderedDict()  #创建有序字典，存储聚类信息
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]  #存储box的索引列表
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # 存坐标 coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs] #存角度

        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw']) #计算角度方差
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres   # 方差是否大于阈值，如果是，则标记角度变化大
        cluster_dict[cluster_id]['active'] = True #当前聚类为有效


        ########### adaptive_landmark ##################
        if landmark_SE2:  #true
            if adaptive_landmark and yaw_var > yaw_var_thres:
                landmark = pred_center_world_cat[box_idx][:2]
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])  #将landmark设置为当前bounding box的中心坐标和角度
                landmark[2] = pred_yaw_world_cat[box_idx]  #更新landmark的角度信息，等于当前bounding box的角度
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
        
        # 递增聚类的ID，准备处理下一个聚类
        cluster_id += 1
        #遍历bounding box索引列表，移除已经分配到聚类中的bounding box
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
        # case1: object num is smaller than 3，如果聚类物体数量小于3，说明物体较少，难以优化位姿，直接返回原始代理的pose信息
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies，如果超过一半的物体角度差异较大，直接返回代理的pose信息，跳出后续的图优化
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:  #如果某物体的角度差异较大，将该物体标记为不活跃，不参与图优化
                cluster_dict[landmark_id]['active'] = False

    """
        开始图优化
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D()  #创建一个二维姿态图用于姿态优化

    # Add agent to vertexs 将代理姿态添加到姿态图中
    for agent_id in range(agent_num):
        v_id = agent_id  #为每个代理分配唯一的id
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]]  #获取代理的位置和朝向
        pose_np[2] = np.deg2rad(pose_np[2])  # radians 将朝向信息从度数转为弧度
        v_pose = g2o.SE2(pose_np)   #创建一个SE2姿态对象
        #如果是第一个代理，将代理的姿态信息添加到图中，并将其标记为固定，它的位姿信息不会变
        # 将其他代理的姿态信息添加到图中，并将其标记为不固定，可以在后续的优化中进行调整
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add object to vertexs 将物体（landmark）的姿态添加到姿态图中
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,) 获取物体的位置朝向
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #判断物体的地标类型是否为SE2
        #如果是SE2，，创建SE2对象，并使用地标信息初始化
        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark
        #将物体的姿态图信息添加到图中，不标记为固定，类型为SE2
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set 将代理与物体之间的关联关系添加到边集
    for landmark_id in range(agent_num, vertex_num):  #遍历所有物体
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']  #获取物体的地标类型
        #如果物体地标类型不是活跃的，跳过，不与之关联
        if not cluster_dict[landmark_id]['active']:
            continue
        #遍历所有与物体关联的框
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]  #找到与bounding box关联的代理id
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64)) #创建一个SE2类型的边，使用框的位置和朝向信息初始化
                info = np.identity(3, dtype=np.float64)  #创建一个3*3的单位矩阵作为信息矩阵，用于优化权重
                if uncertainty_list is not None:  #如果提供了不确定性信息
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx] #根据不确定信息更新信息矩阵的对角线元素

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:  #如果不确定性总和小于100，跳过这个边
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue
            #将边信息添加到姿态图中，表示代理和物体之间的关联
            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)  #对姿态图进行优化，最大化迭代次数

    pose_new_list = []  #创建一个空列表，用于存储优化后的姿态信息
    for agent_id in range(agent_num):  #遍历所有代理
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector()) #获取并添加每个代理优化后的姿态信息

    refined_pose = np.array(pose_new_list)  #将优化后的姿态信息转换为Numpy数组
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source #将姿态信息的朝向从弧度转为度数

    return refined_pose  #返回优化后的姿态信息，包括位置和朝向

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


