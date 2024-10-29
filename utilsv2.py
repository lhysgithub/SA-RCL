import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

class PreDataset(Dataset):
    def __init__(self, raw_data,args):
        self.data = raw_data
        self.args = args
    
    def __getitem__(self, index):
        x = self.data[index:index+self.args.win_size]
        y = self.data[index+self.args.win_size:index+self.args.win_size+self.args.pred_win_size]
        return x,y
    
    def __len__(self):
        return len(self.data) - self.args.win_size - self.args.pred_win_size + 1
    
class PreTestDataset(Dataset):
    def __init__(self, raw_data,args):
        self.data = raw_data
        self.args = args
    
    def __getitem__(self, index):
        index = index*self.args.win_size
        x = self.data[index:index+self.args.win_size]
        y = self.data[index+self.args.win_size:index+self.args.win_size+self.args.pred_win_size]
        return x,y
    
    def __len__(self):
        return int((len(self.data) - 1 ) / self.args.win_size )

# compute constraint h(A) value
def h_A(A, m):
    expm_A = torch.matrix_power(torch.eye(len(A)).to(A.device)+A*A, m)
    return torch.trace(expm_A) - len(A)

def get_index(names,useless_metric_name,args):
    useless_metric_index = []
    for n in names:
        if n in useless_metric_name:
            useless_metric_index.append(1)
        else:
            useless_metric_index.append(0)
    useless_metric_index = torch.Tensor(np.array(useless_metric_index)).to(args.device)
    return useless_metric_index

# def kl_gaussian_sem(preds):
#     mu = preds
#     kl_div = mu * mu
#     kl_sum = kl_div.sum()
#     return (kl_sum / (preds.size(0)))*0.5

def evaluate_dfs(graph,names,root_name,args,epoch,case_number):
    # th = np.percentile(graph, args.graph_edge_th)
    graph_ = graph.reshape(-1)
    graph_edge_th = min(args.graph_edge_th,len(graph))
    topk_number = int(len(graph)*graph_edge_th)
    topk_index = np.argpartition(graph_,len(graph_)-topk_number)
    graph__ = graph_[topk_index[len(graph_)-topk_number:]]
    try:
        th = graph__.min()
    except Exception as e:
        print("graph is empty") 
        th = 0 
    graph[graph <= th] = 0
    graph[graph < 0] = 0
    
    # 可视化Graph # todo debug cause modelling
    if args.debug:
        # if epoch!=0 and epoch % (args.epochs-1) ==0 :
        if epoch % (10) == 0 :
            org_G = nx.from_numpy_matrix(graph, parallel_edges=True, create_using=nx.DiGraph)
            name_G = nx.DiGraph()  # 创建空有向图
            # columns = [i.split('_')[0] for i in columns]
            for i in range(len(names)):
                name_G.add_node(names[i])
            for i in list(org_G.edges):
                if graph[i[0],i[1]] >0:
                    name_G.add_edge(names[i[0]],names[i[1]])
            pos=nx.circular_layout(name_G)
            plt.cla()
            nx.draw(name_G, pos=pos, with_labels=True)
            plt.savefig(f"analysis/metrics_causality_{case_number}_{epoch}.png")
    
    

def evaluate(graph,names,root_name,args,epoch,case_number):
    # th = np.percentile(graph, args.graph_edge_th)
    graph_ = graph.reshape(-1)
    graph_edge_th = min(args.graph_edge_th,len(graph))
    topk_number = int(len(graph)*graph_edge_th)
    topk_index = np.argpartition(graph_,len(graph_)-topk_number)
    graph__ = graph_[topk_index[len(graph_)-topk_number:]]
    try:
        th = graph__.min()
    except Exception as e:
        print("graph is empty") 
        th = 0 
    graph[graph <= th] = 0
    graph[graph < 0] = 0
    
    # 可视化Graph # todo debug cause modelling
    if args.debug:
        # if epoch!=0 and epoch % (args.epochs-1) ==0 :
        if epoch % (10) == 0 :
            org_G = nx.from_numpy_matrix(graph, parallel_edges=True, create_using=nx.DiGraph)
            name_G = nx.DiGraph()  # 创建空有向图
            # columns = [i.split('_')[0] for i in columns]
            for i in range(len(names)):
                name_G.add_node(names[i])
            for i in list(org_G.edges):
                if graph[i[0],i[1]] >0:
                    name_G.add_edge(names[i[0]],names[i[1]])
            pos=nx.circular_layout(name_G)
            plt.cla()
            nx.draw(name_G, pos=pos, with_labels=True)
            plt.savefig(f"analysis/metrics_causality_{case_number}_{epoch}.png")
    
    
    # PageRank
    from sknetwork.ranking import PageRank
    pagerank = PageRank()
    try:
        scores = pagerank.fit(np.abs(graph.T)).scores_
        # print("graph is not empty")
    except:
        print("graph is empty") # todo fix too many situation graph is empty
        scores = np.ones(graph.shape[0])

    # recording for evaluation
    score_dict = {}
    for i,s in enumerate(scores):
        score_dict[names[i]] = s
    sorted_dict = sorted(score_dict.items(), key=lambda item:item[1], reverse=True)
    rank_names = [i[0] for i in sorted_dict]
    hits = np.zeros(args.evaluate_k+1)
    hits_sum =0
    # for rn in root_name:
    for i in range(args.evaluate_k):
        for j in range(i):
            if j >= len(rank_names):
                continue
            if rank_names[j] in root_name:
                hits[i+1] += 1
        hits_sum += hits[i+1]
    return scores, hits_sum


def drop_coliner_data(raw_data,coliner_theta=0.01):
    drop_list = []
    columns = list(raw_data)
    stds = []
    for i in columns:
        ser = raw_data.loc[:,i]
        std = ser.std()
        stds.append(std)
        if std <= coliner_theta:
            drop_list.append(i)
    data = raw_data.drop(drop_list, axis=1)
    print(f"drop fixedly dims: {len(drop_list)}")
    return data

# import ruptures as rpt
# def state_segment(observation_data,args):
#     algo = rpt.Window(model="l2", min_size=3, jump=3).fit(observation_data)
#     result = algo.predict(n_bkps=1)
#     changepoint = result[-2] if len(result)>1 else result[-1]
#     observation_data_np = observation_data.to_numpy()
#     observation_data_np = observation_data_np[changepoint:]
#     observation_data = pd.DataFrame(observation_data_np,columns=observation_data.columns)
#     return observation_data

# def cost_func_Sigma(data):
#     mu = data.mean(axis=0)
#     std = data.std(axis=0)
#     data2 = np.abs(data - mu)
#     std2 = data2.std(axis=0)
#     cost = len(data)*(std**2) + (data2/(std2**2)).sum(axis=0)
#     return cost

# def cost_func_Sigma_v2(data):
#     mu = data.mean(axis=0)
#     std = data.std(axis=0)
#     data2 = np.abs(data - mu)
#     std2 = data2.std(axis=0)
#     cost = len(data)*(std**2) + len(data)*(std2)
#     return cost

# def cost_func_l2_v2(data):
#     mu = data.mean(axis=0)
#     data2 = data - mu
#     cost =  (data2**2).sum(axis=0)
#     return cost

# def cost_func_l2_v1(data):
#     std = data.std(axis=0)
#     cost = len(data)*(std**2)
#     return cost

# def change_detection(observation_data,anomaly_data,args):
#     cat_observation_data = pd.concat([observation_data,anomaly_data]).to_numpy()
#     observation_len = len(observation_data)
#     anomaly_len = len(anomaly_data)
#     cat_len = len(cat_observation_data)
#     variates = list(observation_data)
#     w = args.win_w
#     c = cost_func_l2_v1
#     # c = cost_func_Sigma_v2
#     # c = cost_func_Sigma
#     data0 = cat_observation_data
#     costs = []
#     for i in range(w*2,cat_len):
#         data1 = data0[i-2*w:i-w]
#         data2 = data0[i-w:i]
#         data3 = data0[i-2*w:i]
#         cost = c(data3)-c(data1)-c(data2)
#         costs.append(cost)
#     costs = np.array(costs)
#     point = observation_len-2*w
#     mu = costs[:point].mean(axis=0)
#     std = costs[:point].std(axis=0)
#     threshold = args.change_detection_theta*std
#     abnormal_costs = costs[point:]
#     is_over_threshold = np.abs(abnormal_costs - mu.reshape(1,-1).repeat(len(abnormal_costs),axis=0)) > threshold.reshape(1,-1).repeat(len(abnormal_costs),axis=0)
#     is_over_threshold = is_over_threshold.astype(int)
#     selected_v = []
#     droped_v = []
#     for j in range(len(variates)):
#         if is_over_threshold[:,j].max() >= 1:
#             selected_v.append(variates[j])
#         else:
#             droped_v.append(variates[j])
#     observation_data = observation_data.loc[:,selected_v]
#     anomaly_data = anomaly_data.loc[:,selected_v]
#     print(f"drop v: {len(droped_v)}  droped_v: {droped_v}")
#     return observation_data,anomaly_data
