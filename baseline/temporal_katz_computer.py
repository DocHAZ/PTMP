import os
import numpy as np
from centrality_utils.base_computer import *
from centrality_utils.weight_funtions import *
from data_processing.tennis_player_processing import load_dataset_parameters
import simulator_utils.graph_simulator as gsim

class TemporalKatzParams():
    def __init__(self,beta,weight_function):
        if beta >= 0 and beta <= 1:
            self.beta = beta
        else:
            raise RuntimeError("'beta' must be from interval [0,1]!")
        self.weight_func = weight_function
        
    def __str__(self):
        return "tk_b%0.2i_%s" % (self.beta,str(self.weight_func))

class TemporalKatzComputer(BaseComputer):
    """General temporal Katz centrality implementation"""
    def __init__(self,nodes,param_list):
        self.param_list = param_list
        self.num_of_nodes = len(nodes)
        self.node_indexes = dict(zip(nodes,range(self.num_of_nodes)))
        self.ranks = np.zeros((self.num_of_nodes,len(self.param_list)))
        self.node_last_activation = {}
        
    def get_updated_node_rank(self,time,node_id):
        node_index = self.node_indexes[node_id]
        updated_ranks = self.ranks[node_index,:] # zero vector if node did not appear before
        if node_id in self.node_last_activation:
            delta_time = time - self.node_last_activation[node_id]
            time_decaying_weights = [param.weight_func.weight(delta_time) for param in self.param_list]
            updated_ranks *= time_decaying_weights
        return node_index, updated_ranks
    
    def get_all_updated_node_ranks(self,time):
        updated_scores = []
        for node in self.node_last_activation:
            node_index, node_rank = self.get_updated_node_rank(time,node)
            row = [node] + list(node_rank)
            updated_scores.append(row)
        return np.array(updated_scores)
    
    def update(self,edge,time,graph=None,snapshot_graph=None):
        src, trg = int(edge[0]), int(edge[1])
        beta_vector = [param.beta for param in self.param_list]
        src_index, src_rank = self.get_updated_node_rank(time,src)
        trg_index, trg_rank = self.get_updated_node_rank(time,trg)
        self.ranks[src_index,:] = src_rank
        self.ranks[trg_index,:] = trg_rank + beta_vector * (src_rank + 1) # +1 is for 1 length path
        self.node_last_activation[src] = time
        self.node_last_activation[trg] = time
        
    def save_snapshot(self,experiment_folder,snapshot_index,time,graph=None,snapshot_graph=None):
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        all_nodes_updated = self.get_all_updated_node_ranks(time)
        for j, param in enumerate(self.param_list):
            output_folder = "%s/%s" % (experiment_folder,param)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            active_arr = all_nodes_updated[:,[0,j+1]]
            scores2file(active_arr,"%s/tk_%i.csv" % (output_folder,snapshot_index))


if __name__ == '__main__':

    dataset_id = "uo17"

    min_epoch, num_days, _, _, _, _ = load_dataset_parameters(dataset_id)
    # print(dataset_id)
    # print('--->')
    delta = 120
    index_threshold = int(num_days * 86520 / delta + 1)
    # print(delta, index_threshold)

    # data_path = '../data/%s_data/raw/%s_mentions.csv' % (dataset_id, dataset_id)
    data_path = '../../data/uo_97/uo_97_125_new.txt'
    # score_output_dir = '../data/%s_data/centrality_measures/' % dataset_id

    score_output_dir = '../../data/uo_97/centrality_measures_tkz_97_125_120/'
    data = np.loadtxt(data_path, delimiter=' ', dtype='i')
    # print('%s dataset were loaded.' % dataset_id)
    # print('Number of edges in data: %i.' % len(data))
    # print(data[:5])
    print(data)

    selector = data[:, 0] >= min_epoch

    data = data[selector, :]
    # print('--------->9')

    gsim_params = []
    tpr_params=[]

    src_unique = np.unique(data[:, 1])
    trg_unique = np.unique(data[:, 2])
    nodes = np.unique(np.concatenate((src_unique, trg_unique)))

    norm_factors = []
    norm_factors += [120.0 * i for i in [1,2,3,4,6,8,10,12,16,32,40,64,88,112,136,272,481,721]]
    # norm_factors += [21600.0 * i for i in [1, 2, 3, 4, 6, 8, 10, 12, 24, 36, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288]]
    print(norm_factors)

    if delta == 120:
        static_lookbacks = [1,2,3,4,6,8,10,12,16,32,40,64,88,112,136,272,481,721]
        # static_lookbacks = [1, 2, 3, 4, 6, 8, 10, 12, 24, 36, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288]
    # else:
    #     static_lookbacks = [0, 1, 2, 4, 7, 14, 21, 30]


    tk_beta = 1.0  # choose beta for temporal Katz centrality

    tk_params = []
    tk_params += [TemporalKatzParams(tk_beta, ExponentialWeighter(base=0.5, norm=n)) for n in norm_factors]
    if len(tk_params) > 0:
        gsim_params.append(TemporalKatzComputer(nodes, tk_params))


    boundaries = min_epoch + np.array([delta * i for i in range(1, index_threshold + 1)])
    gsim_obj = gsim.OnlineGraphSimulator(data, time_type="epoch", verbose=True)

    nexperiment_graph_stats = gsim_obj.run_with_boundaries(gsim_params, boundaries, score_output_dir,
                                                           max_index=index_threshold)
        
#
# class TruncatedTemporalKatzParams():
#     def __init__(self,beta,weight_function):
#         if beta >= 0 and beta <= 1:
#             self.beta = beta
#         else:
#             raise RuntimeError("'beta' must be from interval [0,1]!")
#         self.weight_func = weight_function
#
#     def __str__(self):
#         return "ttk_b%0.2f_%s" % (self.beta,str(self.weight_func))
#
# class TruncatedTemporalKatzComputer(BaseComputer):
#     """Truncated temporal Katz centrality implementation"""
#     def __init__(self,nodes,param_list,k=5):
#         self.k = k
#         self.param_list = param_list
#         self.num_of_nodes = len(nodes)
#         self.node_indexes = dict(zip(nodes,range(self.num_of_nodes)))
#         self.ranks = [np.zeros((self.num_of_nodes,len(self.param_list))) for i in range(k)]
#         self.beta_vector = [param.beta for param in self.param_list]
#         self.node_last_activation = {}
#
#     def get_updated_node_rank(self,layer_idx,time,node_id):
#         node_index = self.node_indexes[node_id]
#         updated_ranks = self.ranks[layer_idx][node_index,:] # zero vector if node did not appear before
#         if node_id in self.node_last_activation:
#             delta_time = time - self.node_last_activation[node_id]
#             time_decaying_weights = [param.weight_func.weight(delta_time) for param in self.param_list]
#             updated_ranks *= time_decaying_weights
#         return node_index, updated_ranks
#
#     def get_all_updated_node_ranks(self,layer_idx,time):
#         updated_scores = []
#         for node in self.node_last_activation:
#             node_index, node_rank = self.get_updated_node_rank(layer_idx,time,node)
#             row = [node] + list(node_rank)
#             updated_scores.append(row)
#         return np.array(updated_scores)
#
#     def update(self,edge,time,graph=None,snapshot_graph=None):
#         src, trg = int(edge[0]), int(edge[1])
#         # update each layer
#         for layer_idx in list(reversed(range(0,self.k))):
#             if layer_idx == 0:
#                 src_rank_shorter = np.zeros(len(self.param_list))
#             else:
#                 _, src_rank_shorter = self.get_updated_node_rank(layer_idx-1,time,src)
#             src_index, src_rank = self.get_updated_node_rank(layer_idx,time,src)
#             trg_index, trg_rank = self.get_updated_node_rank(layer_idx,time,trg)
#             self.ranks[layer_idx][src_index,:] = src_rank
#             self.ranks[layer_idx][trg_index,:] = trg_rank + self.beta_vector * (src_rank_shorter + 1) # +1 is for 1 length path
#         self.node_last_activation[src] = time
#         self.node_last_activation[trg] = time
#
#     def save_snapshot(self,experiment_folder,snapshot_index,time,graph,snapshot_graph=None):
#         """Exports every truncated score with maximum length from 1 to k"""
#         if not os.path.exists(experiment_folder):
#             os.makedirs(experiment_folder)
#         for layer_idx in list(reversed(range(0,self.k))):
#             all_nodes_updated = self.get_all_updated_node_ranks(layer_idx,time)
#             for j, param in enumerate(self.param_list):
#                 output_folder = "%s/%s_length_limit_%i" % (experiment_folder,param,layer_idx+1)
#                 if not os.path.exists(output_folder):
#                     os.makedirs(output_folder)
#                 active_arr = all_nodes_updated[:,[0,j+1]]
#                 scores2file(active_arr,"%s/ttk_%i.csv" % (output_folder,snapshot_index))