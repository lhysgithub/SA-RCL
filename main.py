import time
import pickle as pkl 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from tqdm import tqdm
from utilsv2 import *
import torch.optim
import os
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import json
from torch.utils.data import Dataset, DataLoader
from RUN import RUN
from CausalRCAv2 import *
from PC import PC
from LiNGAM import LiNGAM
from GES import GES

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_metrics(data,case_number,root_name):
    plt.cla()
    m = list(data.columns)
    plt.figure(figsize=(13,10))
    for each in m:
        x = np.arange(len(data[each]))
        if each in root_name:
            plt.plot(x,data[each], label=str(each))
        else:
            plt.plot(x,data[each], alpha=0.1)
    plt.legend()
    plt.savefig(f"analysis/metrics_visualization_{case_number}_{root_name}.pdf")

class CDataset(Dataset):
    def __init__(self, raw_data,args):
        self.data = raw_data
        self.args = args
    
    def __getitem__(self, index):
        x = self.data[index:index+self.args.min_state_win_size]
        y = self.data[index+self.args.min_state_win_size:index+self.args.min_state_win_size+self.args.min_state_win_size]
        return x,y
    
    def __len__(self):
        return len(self.data) - self.args.min_state_win_size - self.args.min_state_win_size + 1 # Note: 注意，双倍窗口最小size若超过了样本size，程序会崩溃

def pre_train(data,args):
    dataset = CDataset(data.to_numpy(),args)
    dataloader = DataLoader(dataset,batch_size=len(dataset),shuffle=True)
    dims = len(list(data))
    model = CModel(args,dims).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.contrastive_lr)
    model.train()

    granger_model1s = []
    granger_model2s = []
    granger_model3s = []
    optimizer1s = []
    optimizer2s = []
    optimizer3s = []
    for epoch in range(args.contrastive_epochs):
        pred_loss_list = []
        contrastive_loss_list = []
        loss_list = []
        contra_posi_loss_list = []
        contra_nega_loss_list = []
        contra_posi_adj_loss_list = []
        contra_nega_adj_loss_list = []
        count = 0
        for x,y in dataloader:
            if epoch == 0:
                adj1 = torch.ones((x.shape[0],x.shape[2],x.shape[2])).to(args.device)*0.5
                adj2 = torch.ones((x.shape[0],x.shape[2],x.shape[2])).to(args.device)*0.5
                adj3 = torch.ones((x.shape[0],x.shape[2],x.shape[2])).to(args.device)*0.5
                granger_model1 = Granger(args,adj1).to(args.device).train()
                granger_model2 = Granger(args,adj2).to(args.device).train()
                granger_model3 = Granger(args,adj3).to(args.device).train()
                optimizer1 = torch.optim.Adam(granger_model1.parameters(),lr=args.contrastive_lr)
                optimizer2 = torch.optim.Adam(granger_model2.parameters(),lr=args.contrastive_lr)
                optimizer3 = torch.optim.Adam(granger_model3.parameters(),lr=args.contrastive_lr)
                granger_model1s.append(granger_model1)
                granger_model2s.append(granger_model2)
                granger_model3s.append(granger_model3)
                optimizer1s.append(optimizer1)
                optimizer2s.append(optimizer2)
                optimizer3s.append(optimizer3)
            count += 1
            granger_model1 = granger_model1s[count-1]
            granger_model2 = granger_model2s[count-1]
            granger_model3 = granger_model3s[count-1]
            optimizer1 = optimizer1s[count-1]
            optimizer2 = optimizer2s[count-1]
            optimizer3 = optimizer3s[count-1]
            
            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            
            yx = torch.concat([y[int(x.shape[0]/2):],y[:int(x.shape[0]/2)]])
            x1, x2, x5, adj1, pred_loss1 = model(x.to(args.device).float(),granger_model1)
            y1, y2, y5, adj2, pred_loss2 = model(y.to(args.device).float(),granger_model2)
            y1_, y2_, y5_, adj3, pred_loss3 = model(yx.to(args.device).float(),granger_model3)
            pred_loss = pred_loss1 + pred_loss2 + pred_loss3 # think: 预测损失主要是用于因果图的学习的，如果不使用因果图，预训练时其实可以不用预测损失。
            
            contra_posi_adj_loss = args.contra_adj_theta*F.mse_loss(adj1,adj2)
            contra_nega_adj_loss = args.contra_adj_theta*F.mse_loss(adj1,adj3)
            
            contra_posi_loss = args.contra_sep_theta*F.mse_loss(x1,y1) + args.contra_joint_theta*F.mse_loss(x2,y2) + contra_posi_adj_loss
            contra_nega_loss = args.contra_sep_theta*F.mse_loss(x1,y1_) + args.contra_joint_theta*F.mse_loss(x2,y2_) + contra_nega_adj_loss # think: 为什么epoch0 时 contra_nega_loss 就很大？
            contrastive_loss = contra_posi_loss + args.contra_theta*torch.max(torch.zeros((1)).to(args.device),contra_posi_loss - contra_nega_loss + args.contra_a*contra_posi_loss) # current 10071412: 看一下修改了对比损失后，对于效果的影响
            # current 10071502：看一下sgc 为何不能再SWAT和WADI数据集上提点……目前看在AIOps上已经达到上一个版本纯隐层特征对比的效果……依然考虑这两个数据集大部分时候都处于平稳状态，是否现有对比方法，难以识别出这样场景下的变更点？依然考虑模型训练的流程
            # contrastive_loss = contra_posi_loss - contra_nega_loss
            
            loss = args.pretrain_recon_theta*pred_loss + contrastive_loss
            # loss = contrastive_loss
            loss.backward()
            optimizer.step()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            
            loss_list.append(loss.item())
            pred_loss_list.append(pred_loss.item())
            contra_posi_loss_list.append(contra_posi_loss.item())
            contrastive_loss_list.append(contrastive_loss.item())
            contra_nega_loss_list.append(contra_nega_loss.item())
            contra_posi_adj_loss_list.append(contra_posi_adj_loss.item())
            contra_nega_adj_loss_list.append(contra_nega_adj_loss.item())
            
        if args.debug:
        # if True:
            print(f"pretrain epoch:{epoch} loss:{np.mean(loss_list)} pred_loss:{np.mean(pred_loss_list)} contrastive_loss:{np.mean(contrastive_loss_list)} contra_posi_loss:{np.mean(contra_posi_loss_list)} contra_nega_loss:{np.mean(contra_nega_loss_list)} contra_posi_adj_loss: {np.mean(contra_posi_adj_loss_list)} contra_nega_adj_loss:{np.mean(contra_nega_adj_loss_list)}")
    return model

def get_joint_contra_score(model,x,y,args):
    for epoch in range(args.contrastive_epochs):
        adj1 = torch.ones((x.shape[0],x.shape[2],x.shape[2])).to(args.device)*0.5
        adj2 = torch.ones((x.shape[0],x.shape[2],x.shape[2])).to(args.device)*0.5
        granger_model1 = Granger(args,adj1).train()
        granger_model2 = Granger(args,adj2).train()
        optimizer1 = torch.optim.Adam(granger_model1.parameters(),lr=args.contrastive_lr)
        optimizer2 = torch.optim.Adam(granger_model2.parameters(),lr=args.contrastive_lr)
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        x1, x2, x5, adj1, pred_loss1 = model(x.to(args.device).float(),granger_model1) # think: 仅一个epoch的训练，因果图可能学的不准，因此现在基于因果图来做变点检测是不合理的
        y1, y2, y5, adj2, pred_loss2 = model(y.to(args.device).float(),granger_model2)
        
        pred_loss = pred_loss1 + pred_loss2
        
        pred_loss.backward()
        optimizer1.step()
        optimizer2.step()
        
    joint_contra_score = args.contra_joint_score_joint_theta*((x2-y2)**2).mean(-1).mean(-1) + args.contra_joint_score_adj_theta*((adj1-adj2)**2).mean(-1).mean(-1)
    return joint_contra_score

def get_long_seperate_representations(model,args,data):
    b,t,d = data.shape
    dataset = PreTestDataset(data.detach().cpu().numpy().reshape(-1,d),args)
    dataloader = DataLoader(dataset,batch_size=len(dataset),shuffle=False)
    adj = torch.ones((d,d)).to(args.device)*0.5
    modelg = Granger2(args, adj).to(args.device).train()
    optimizerg = optim.Adam(modelg.parameters(),lr=args.lr)
    for epoch in range(args.contrastive_epochs):
        representations = []
        loss_list = []
        for x,y in dataloader:
            optimizerg.zero_grad()
            x = x.to(args.device)
            target = y.to(args.device)
            x1, _, output, origin_A, _ = model(x,modelg,False)
            pred_loss = F.mse_loss(output, target)
            pred_loss.backward()
            optimizerg.step()
            representations.append(x1.detach().cpu().numpy())
            loss_list.append(pred_loss.item())
        if args.debug:
            print(f"get_long_seperate_representations epoch:{epoch} pred_loss:{np.mean(loss_list)}")
    representations = np.concatenate(representations,axis=0).mean(0) # current 10071925：看一下修改了对比分立分数计算方式后，Remove_before 和 End2End的表现。运气好就直接出来了，运气不好仍需调优End2End的表现。
    return representations

def get_seperate_contra_score(model,x,y,args):
    x1 = get_long_seperate_representations(model,args,x)
    y1 = get_long_seperate_representations(model,args,y)
    sep_contra_score = ((x1-y1)**2).mean(-2)
    return sep_contra_score

def segmentation(data,smodel,args,fault_occur_time):
    names = list(data)
    dataset = CDataset(data,args)
    dataloader = DataLoader(dataset,batch_size=args.contrastive_batch_size,shuffle=False) # 此处注意，难以检测第一个win_size内的状态变更，以及最后一个win_size内的状态变化
    joint_contra_loss_list = []
    loss_list = []
    loss_fn = get_l1_loss
    if args.Segmentation == "L1":
        loss_fn = get_l1_loss
    elif args.Segmentation == "L2": # lhy todo check Segmentation is useless
        loss_fn = get_l2_loss
    elif args.Segmentation == "Sigma":
        loss_fn = get_sigma_loss 
    for x,y in dataloader:
        if args.Segmentation == "Contrastive":
            joint_contra_score = get_joint_contra_score(smodel,x.to(args.device).float(),y.to(args.device).float(),args)
            # _, _,joint_contra_score,_ = smodel(x.to(args.device).float(),y.to(args.device).float()) # joint_contra_score 由于数据集构造的问题，此处检测CP的初始点是args.win_size，但其下标为0
            joint_contra_loss_list.append(joint_contra_score.detach().cpu().numpy())
        else:
            loss_list.append(loss_fn(x,y)[0].numpy())
    
    if args.Segmentation == "Contrastive":
        change_point,occur_time,latest_normal_start,recent_abnormal_end,occur_time2 = find_change_point(data,np.concatenate(joint_contra_loss_list),args)
    else:
        change_point,occur_time,latest_normal_start,recent_abnormal_end,occur_time2 = find_change_point(data,np.concatenate(loss_list),args)
    if len(change_point) == 0:
        return data,[""]
    
    print(f"change_point:{change_point} occur_time:{occur_time} latest_normal_start:{latest_normal_start} recent_abnormal_end:{recent_abnormal_end} occur_time2:{occur_time2}")
    
    # 直接根据故障发生点，前后进行状态变化检测，用于识别故障无关指标
    x_ = data[occur_time:occur_time + args.min_state_win_size].to_numpy().reshape(1,args.min_state_win_size,-1) # todo fix cao, 变更点检测，还需要检测到分布发生变化，现在的L1和L2都不能检测到分布发生变化
    y_ = data[occur_time+ args.min_state_win_size:occur_time + args.min_state_win_size + args.min_state_win_size].to_numpy().reshape(1,args.min_state_win_size,-1)
    # _, _,_,seperate_contra_score = smodel(torch.Tensor(x_).to(args.device).float(),torch.Tensor(y_).to(args.device).float())
    if args.Segmentation == "Contrastive":
        seperate_contra_score = get_seperate_contra_score(smodel,torch.Tensor(x_).to(args.device).float(),torch.Tensor(y_).to(args.device).float(),args)
        seperate_contra_score = seperate_contra_score.detach().cpu().numpy().reshape(-1)
    else:
        seperate_contra_score = loss_fn(torch.Tensor(x_).float(),torch.Tensor(y_).float())[1].numpy().reshape(-1)
    th2 = np.percentile(seperate_contra_score,args.useless_metric_th) # useless_metric_th
    useless_metric_index = np.where(seperate_contra_score<th2)
    useless_metric_name = np.array(names)[useless_metric_index]
    useless_metric_names = []
    for i in useless_metric_name:
        useless_metric_names.append(i)
    
    # Normal Abnormal All
    
    if args.used_state == "All":
        output_data = data[args.min_state_win_size+latest_normal_start:args.min_state_win_size+recent_abnormal_end]
    elif args.used_state == "Normal":
        output_data = data[args.min_state_win_size+latest_normal_start:args.min_state_win_size+occur_time2]
    elif args.used_state == "Abnormal":
        output_data = data[args.min_state_win_size+occur_time2:args.min_state_win_size+recent_abnormal_end]
    elif args.used_state == "None":
        output_data = data
    return output_data,useless_metric_names

def segmentationv2(data,smodel,args,fault_occur_time):
    names = list(data)
    dataset = CDataset(data.to_numpy(),args)
    dataloader = DataLoader(dataset,batch_size=len(dataset),shuffle=False) # 此处注意，难以检测第一个win_size内的状态变更，以及最后一个win_size内的状态变化
    
    loss_list = []
    loss_fn = get_l2_loss
    if args.Segmentation == "L1":
        loss_fn = get_l1_loss
    elif args.Segmentation == "L2": # lhy todo check Segmentation is useless
        loss_fn = get_l2_loss
    elif args.Segmentation == "Sigma":
        loss_fn = get_sigma_loss 
        
    joint_contra_loss_list = []
    for x,y in dataloader:
        if args.Segmentation == "Contrastive":
            joint_contra_score = get_joint_contra_score(smodel,x.to(args.device).float(),y.to(args.device).float(),args)
            # _, _,joint_contra_score,_ = smodel(x.to(args.device).float(),y.to(args.device).float()) # joint_contra_score 由于数据集构造的问题，此处检测CP的初始点是args.win_size，但其下标为0
            joint_contra_loss_list.append(joint_contra_score.detach().cpu().numpy())
        else:
            loss_list.append(loss_fn(x,y)[0].numpy())
    
    if args.Segmentation == "Contrastive":
        change_point,occur_time,latest_normal_start,future_abnormal_end = find_change_pointv2(data,np.concatenate(joint_contra_loss_list),args,fault_occur_time)
    else:
        change_point,occur_time,latest_normal_start,future_abnormal_end = find_change_pointv2(data,np.concatenate(loss_list),args,fault_occur_time)
    if len(change_point) == 0:
        return data,fault_occur_time
    
    print("")
    print(f"change_point:{change_point} occur_time:{occur_time} latest_normal_start:{latest_normal_start} future_abnormal_end:{future_abnormal_end}")
    
    # 直接根据故障发生点，前后进行状态变化检测，用于识别故障无关指标
    # occur_time2 = (latest_normal_start+future_abnormal_end)/2
    
    # Normal Abnormal All
    if args.used_state == "All":
        output_data = data.iloc[latest_normal_start:future_abnormal_end,:]
        occur_time = fault_occur_time - latest_normal_start
    elif args.used_state == "Normal":
        output_data = data.iloc[latest_normal_start:,:]
        occur_time = fault_occur_time - latest_normal_start
    elif args.used_state == "Abnormal":
        output_data = data.iloc[:future_abnormal_end,:]
        occur_time = fault_occur_time
    elif args.used_state == "None":
        output_data = data
        occur_time = fault_occur_time
    return output_data,occur_time

def useless_filter(data,smodel,args,occur_time):
    names = list(data)
    
    loss_fn = get_l2_loss
    if args.Segmentation == "L1":
        loss_fn = get_l1_loss
    elif args.Segmentation == "L2": # lhy todo check Segmentation is useless
        loss_fn = get_l2_loss
    elif args.Segmentation == "Sigma":
        loss_fn = get_sigma_loss 
    
    if args.filter_scope == "all":
        x_ = data.iloc[:occur_time,:].to_numpy().reshape(1,-1,len(list(data))) # todo fix cao, 变更点检测，还需要检测到分布发生变化，现在的L1和L2都不能检测到分布发生变化
        y_ = data.iloc[occur_time:,:].to_numpy().reshape(1,-1,len(list(data)))
    else: # local
        if occur_time-args.min_state_win_size < 0:
            args.to_samll_contra += 1
            latest_normal_start2 = 0
            x_ = data.iloc[latest_normal_start2:latest_normal_start2+args.min_state_win_size,:].to_numpy().reshape(1,args.min_state_win_size,-1) # todo fix cao, 变更点检测，还需要检测到分布发生变化，现在的L1和L2都不能检测到分布发生变化
            y_ = data.iloc[latest_normal_start2+args.min_state_win_size:latest_normal_start2 + args.min_state_win_size*2,:].to_numpy().reshape(1,args.min_state_win_size,-1)
        elif occur_time+args.min_state_win_size > len(data)-1:
            args.to_samll_contra += 1
            future_abnormal_end2 = len(data)-1
            x_ = data.iloc[future_abnormal_end2-args.min_state_win_size*2:future_abnormal_end2-args.min_state_win_size,:].to_numpy().reshape(1,args.min_state_win_size,-1) # todo fix cao, 变更点检测，还需要检测到分布发生变化，现在的L1和L2都不能检测到分布发生变化
            y_ = data.iloc[future_abnormal_end2-args.min_state_win_size:future_abnormal_end2,:].to_numpy().reshape(1,args.min_state_win_size,-1)
        else:
            x_ = data.iloc[occur_time-args.min_state_win_size:occur_time,:].to_numpy().reshape(1,args.min_state_win_size,-1) # todo fix cao, 变更点检测，还需要检测到分布发生变化，现在的L1和L2都不能检测到分布发生变化
            y_ = data.iloc[occur_time:occur_time+args.min_state_win_size,:].to_numpy().reshape(1,args.min_state_win_size,-1)
    
    # _, _,_,seperate_contra_score = smodel(torch.Tensor(x_).to(args.device).float(),torch.Tensor(y_).to(args.device).float())
    if args.Segmentation == "Contrastive": # todo fix tmd 此处需要能够接受变长输入
        try:
            seperate_contra_score = get_seperate_contra_score(smodel,torch.Tensor(x_).to(args.device).float(),torch.Tensor(y_).to(args.device).float(),args)
            seperate_contra_score = seperate_contra_score.reshape(-1)
        except Exception as e:
            args.to_samll_sep += 1
            print("too small x_ or y_")
            seperate_contra_score = loss_fn(torch.Tensor(x_).float(),torch.Tensor(y_).float())[1].numpy().reshape(-1)
    else:
        seperate_contra_score = loss_fn(torch.Tensor(x_).float(),torch.Tensor(y_).float())[1].numpy().reshape(-1)
    th2 = np.percentile(seperate_contra_score,args.useless_metric_th) # useless_metric_th
    if args.reverse_filter:
        useless_metric_index = np.where(seperate_contra_score>th2)
    else:
        useless_metric_index = np.where(seperate_contra_score<th2)
    useless_metric_name = np.array(names)[useless_metric_index]
    useless_metric_names = []
    for i in useless_metric_name:
        useless_metric_names.append(i)
        
    return useless_metric_names

def get_l1_loss(x,y):
    c = torch.concat([x,y],dim=1)
    
    b,t,d = x.shape
    _,t2,_ = y.shape
    _,t3,_ = c.shape
    
    mean1 = x.mean(-2)
    std1 = x.std(-2)
    
    mean2 = y.mean(-2)
    std2 = y.std(-2)
    
    mean3 = c.mean(-2)
    std3 = c.std(-2)
    
    l1_loss_t = torch.abs(c - mean3.unsqueeze(1).repeat(1,t3,1)).mean(-1).mean(-1) - torch.abs(x - mean1.unsqueeze(1).repeat(1,t,1)).mean(-1).mean(-1) - torch.abs(y - mean2.unsqueeze(1).repeat(1,t2,1)).mean(-1).mean(-1) 
    l1_loss_d = torch.abs(c - mean3.unsqueeze(1).repeat(1,t3,1)).mean(-2) - torch.abs(x - mean1.unsqueeze(1).repeat(1,t,1)).mean(-2) - torch.abs(y - mean2.unsqueeze(1).repeat(1,t2,1)).mean(-2) 
    return l1_loss_t,l1_loss_d

def get_l2_loss(x,y):
    c = torch.concat([x,y],dim=1)
    
    b,t,d = x.shape
    _,t2,_ = y.shape
    _,t3,_ = c.shape
    
    mean1 = x.mean(-2)
    std1 = x.std(-2)
    
    mean2 = y.mean(-2)
    std2 = y.std(-2)
    
    mean3 = c.mean(-2)
    std3 = c.std(-2)
    
    l2_loss_t = ((c - mean3.unsqueeze(1).repeat(1,t3,1))**2).mean(-1).mean(-1) - ((x - mean1.unsqueeze(1).repeat(1,t,1))**2).mean(-1).mean(-1) - ((y - mean2.unsqueeze(1).repeat(1,t2,1))**2).mean(-1).mean(-1)
    l2_loss_d = ((c - mean3.unsqueeze(1).repeat(1,t3,1))**2).mean(-2) - ((x - mean1.unsqueeze(1).repeat(1,t,1))**2).mean(-2) - ((y - mean2.unsqueeze(1).repeat(1,t2,1))**2).mean(-2)
    
    return l2_loss_t,l2_loss_d

def get_sigma_loss(x,y):
    
    c = torch.concat([x,y],dim=1)
    
    b,t,d = x.shape
    _,t2,_ = y.shape
    _,t3,_ = c.shape
    
    mean1 = x.mean(-2)
    std1 = x.std(-2)
    
    mean2 = y.mean(-2)
    std2 = y.std(-2)
    
    mean3 = c.mean(-2)
    std3 = c.std(-2)
    
    sigma_loss_t = (torch.abs(c - mean3.unsqueeze(1).repeat(1,t3,1))/(std3.unsqueeze(1).repeat(1,t3,1)+1e-5)).mean(-1).mean(-1) - (torch.abs(x - mean1.unsqueeze(1).repeat(1,t,1))/(std1.unsqueeze(1).repeat(1,t,1)+1e-5)).mean(-1).mean(-1) - (torch.abs(y - mean2.unsqueeze(1).repeat(1,t2,1))/(std2.unsqueeze(1).repeat(1,t2,1)+1e-5)).mean(-1).mean(-1)
    sigma_loss_d = (torch.abs(c - mean3.unsqueeze(1).repeat(1,t3,1))/(std3.unsqueeze(1).repeat(1,t3,1)+1e-5)).mean(-2) - (torch.abs(x - mean1.unsqueeze(1).repeat(1,t,1))/(std1.unsqueeze(1).repeat(1,t,1)+1e-5)).mean(-2) - (torch.abs(y - mean2.unsqueeze(1).repeat(1,t2,1))/(std2.unsqueeze(1).repeat(1,t2,1)+1e-5)).mean(-2) 
    
    return sigma_loss_t,sigma_loss_d

def find_change_point(data,loss_list,args):
    # 检测故障发生点，即joint_contra_loss最大的点
    occur_time = np.argmax(loss_list)
    th = np.percentile(loss_list,args.change_point_th) # change_point_th
    change_point = np.where(loss_list > th)[0] # todo done 此处要设计一个最小状态窗口长度，不然可能一个状态就一个timestamp
    if len(change_point) == 0:
        return change_point,0,0,0,0
    
    occur_time_index = np.where(change_point == occur_time)[0][0]
    if occur_time_index==0: # 再仔细想想
        latest_normal_start = 0 - args.min_state_win_size# 此处有个bug
    else:
        latest_normal_start = change_point[occur_time_index-1] # if occur_time_index >= 1 
    if occur_time - latest_normal_start < args.min_state_win_size:
        latest_normal_start = occur_time - args.min_state_win_size - 1 # 如果这个数小于0呢？

    if occur_time_index == len(change_point)-1:
        recent_abnormal_end = len(data) - args.min_state_win_size
    else:
        recent_abnormal_end = change_point[occur_time_index+1]
    if recent_abnormal_end - occur_time < args.min_state_win_size:
        recent_abnormal_end = occur_time + args.min_state_win_size +1
    
    occur_time2 = occur_time
    if latest_normal_start < - args.min_state_win_size:
        recent_abnormal_end = recent_abnormal_end + (-latest_normal_start - args.min_state_win_size)
        occur_time2 = occur_time + (-latest_normal_start - args.min_state_win_size)
        latest_normal_start = - args.min_state_win_size
        
    if recent_abnormal_end > len(data) - args.min_state_win_size:
        latest_normal_start = latest_normal_start - (recent_abnormal_end - len(data) + args.min_state_win_size) 
        occur_time2 = occur_time - (recent_abnormal_end - len(data) + args.min_state_win_size) 
        recent_abnormal_end = len(data) - args.min_state_win_size
        
    return change_point,occur_time,latest_normal_start,recent_abnormal_end,occur_time2

def find_change_pointv2(data,loss_list,args,fault_occur_time):
    # 检测故障发生点，即joint_contra_loss最大的点
    # occur_time = np.argmax(loss_list)
    
    th = np.percentile(loss_list,args.change_point_th) # change_point_th
    change_point = np.where(loss_list > th)[0] # todo done 此处要设计一个最小状态窗口长度，不然可能一个状态就一个timestamp # 此处有bug，真正的故障发生点可能没超过变更点的阈值
    if len(change_point) == 0:
        return change_point,0,0,0
    
    change_point = change_point + np.ones_like(change_point)*args.min_state_win_size
    last_change_point = change_point[change_point<fault_occur_time]
    future_change_point = change_point[change_point>fault_occur_time]
    
    if len(last_change_point) <= 0:
        latest_normal_start = 0
        args.last_nochange_count +=  1
    else:
        latest_normal_start = last_change_point[-1]

    if len(future_change_point) <= 0:
        future_abnormal_end = len(data) - 1
        args.future_nochange_count +=  1
    else:
        future_abnormal_end = future_change_point[0]
        
    if future_abnormal_end - latest_normal_start < args.min_state_win_size:
        args.to_samll_sega +=1
        if latest_normal_start == 0:
            future_abnormal_end = args.latest_normal_start + args.min_state_win_size
        elif future_abnormal_end == len(data)-1:
            latest_normal_start = future_abnormal_end - args.min_state_win_size
        else:
            latest_normal_start = fault_occur_time - int(0.5*args.min_state_win_size)
            future_abnormal_end = fault_occur_time + int(0.5*args.min_state_win_size)
    
    return change_point,fault_occur_time,latest_normal_start,future_abnormal_end
     
# 主要依托segmentation部分的设计
def drop_redundancy_data(data,useless_metric_names):
    wait_to_drop = []
    for name in useless_metric_names:
        if name in list(data):
            wait_to_drop.append(name)
    useless_metric_names = wait_to_drop
    data = data.drop(useless_metric_names, axis=1)
    print(f"drop redundancy dims: {len(useless_metric_names)}")
    return data
       
def drop_redundancy_grahp(graph,useless_metric_names,names,args):
    drop_index = []
    for i in useless_metric_names:
        if i in names:
            drop_index.append(names.index(i))
    
    graph1 = graph
    graph1 = np.delete(graph1,drop_index,axis=0)
    graph1 = np.delete(graph1,drop_index,axis=1)
    
    return graph1

def reweight_redundancy_grahp(graph,useless_metric_names,names,args):
    drop_index = []
    for i in useless_metric_names:
        if i in names:
            drop_index.append(names.index(i))
    
    graph1 = graph
    graph1[drop_index,:] = graph1[drop_index,:]*args.reweight
    graph1[:,drop_index] = graph1[:,drop_index]*args.reweight
    
    return graph1

    # AIOps_db 不同case 采集的指标数量不一样……
    # AIOps_CPU 去除稳定指标后，就只剩一个指标了……
    # AllMetrics_all 不同case采集的指标数量也可能不一样……
    # 同一个数据集不同case采集指标数量不一致的问题，可以考虑针对每一个case预训练一个Model，消融时可以深入讨论一下……

def main():
    parser = argparse.ArgumentParser()
    
    # environment setting
    parser.add_argument('--gpu_device', type=str, default="1", help='')
    parser.add_argument('--random_seed', type=int, default=42, help='')
    parser.add_argument('--gpu', type=str2bool, default=True, help='')
    parser.add_argument('--evaluate_k', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--epochs',type=int, default=51, help='') # current 10081122：51貌似有更好的效果，重跑消融和调参实验
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--check_saved',type=int,default=0,help="")
    parser.add_argument('--debug',type=int,default=0,help="")
    parser.add_argument('--debug_case',type=int,default=1,help="")
    
    # dataset and model setting
    parser.add_argument('--RCA_Model', type=str, default="CausalRCA", help='') # RUN PC LiNGAM CausalRCA # GES 
    parser.add_argument('--data_name', type=str, default="AIOps_all", help='')  # SWAT AIOps_all WADI

    # base model
    parser.add_argument('--sparse_loss',type=int,default=1,help="")
    parser.add_argument('--hid_dims',type=int, default=16, help='') 
    parser.add_argument('--graph_edge_th',type=float,default=2.0,help="") 
    parser.add_argument('--nocycle_order',type=int,default=2,help="")
    parser.add_argument('--win_size', type=int, default=20, help='') # win_size = 20 is best  # 基于对比的变点检测需要大窗口才能捕捉状态的变化，可以考虑直接给对比学习设计一个窗口
    parser.add_argument('--pred_win_size', type=int, default=2, help='')
    
    # segmentation setting
    parser.add_argument('--Segmentation', type=str, default="Contrastive", help='') # None Contrastive L1 L2 Sigma 
    parser.add_argument('--change_point_th',type=float,default=98.0,help="") # 参数敏感实验 [95,90,85,80,75] # current 10071436: 尝试浮点数级别的阈值 # current 10080914: us=none时，看看l2条件下，ct对于过滤的影响。 # think: ct对于过滤到底有没有影响？应该是有的，等等，好像没有，过滤不需要状态的划分，自然也就不需要在时间维度上找变点，也就用不到ct，所以ct对于过滤是完全没有影响的！
    parser.add_argument('--used_state',type=str,default="All",help="") # Normal Abnormal All None # current 10080758: 看看改成None效果怎么样
    # parser.add_argument('--use_',type=str,default="All",help="")
    
    # contrastive model
    parser.add_argument('--contrastive_batch_size', type=int, default=16, help='') # 现在未使用 [8,16,32,64,128] # think: cbs的大小直接影响对比学习时正负样本的生成，在深想一步，对于WADI这种大部分时刻指标都很平稳的数据集，一个cbs内可能全部指标不会发生变化，此时正负样本可能长得一样，相反的优化方向反而限制了基于对比的变点检测。
    parser.add_argument('--contrastive_lr', type=float, default=1e-3, help='') 
    parser.add_argument('--contrastive_epochs',type=int, default=30, help='') # [5,10,15,20,25]
    parser.add_argument('--contra_a',type=float,default=3.0,help="") # [1,2,3,4,5]
    parser.add_argument('--contrastive_hid_dims',type=int, default=64, help='') # [8,16,32,64,128]
    parser.add_argument('--contra_theta',type=float, default=2.0, help='')
    parser.add_argument('--contra_joint_theta',type=float, default=1.0, help='')
    parser.add_argument('--contra_sep_theta',type=float, default=1.0, help='')
    parser.add_argument('--contra_adj_theta',type=float, default=3000.0, help='') # current 10091909: 探讨下at对于效果的影响
    parser.add_argument('--pretrain_recon_theta',type=float, default=1.0, help='')
    parser.add_argument('--contra_joint_score_adj_theta',type=float, default=3000.0, help='')
    parser.add_argument('--contra_joint_score_joint_theta',type=float, default=1.0, help='')
    
    # useless_metric filtering setting 
    parser.add_argument('--Filtering_nodes', type=str, default="End2End", help='') # None Remove_before Remove_after End2End Reweight # Reweight 可能是无效的
    parser.add_argument('--reweight',type=float,default=0.1,help="") # [0.001,0.005,0.01,0.05,0.1,0.5]
    parser.add_argument('--useless_metric_th',type=float,default=92.0,help="") # 参数敏感实验 [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95] # current 10080910: us=none时，看看Contrastive条件下，ut对于过滤的影响。
    parser.add_argument('--filter_theta',type=float,default=1.0,help="") # 还没想清楚怎么做 current 10072215: 看看调整无用指标的训练权重，是否能够改善e2e在SWAT和WADI上的表现
    parser.add_argument('--reverse_filter',type=int,default=0,help="")
    parser.add_argument('--filter_th',type=float,default=0.0,help="")
    parser.add_argument('--filter_scope',type=str,default="all",help="") # local all
    parser.add_argument('--filter_first',type=int,default=0,help="")
    # parser.add_argument('--end2end_after_filter',type=int,default=1,help="") #
    
    args = parser.parse_args()
    args.min_state_win_size = args.win_size + args.pred_win_size
    args.data_path = f"{args.data_name}.pkl"
    args.to_samll_sega = 0
    args.last_nochange_count = 0
    args.future_nochange_count = 0
    args.to_samll_contra = 0
    args.to_samll_sep= 0
    
    if args.data_name == "AIOps_all":
        args.end2end_after_filter = 0 
    else:
        args.end2end_after_filter = 1
    
    
    # 确定RCA Model
    if args.RCA_Model == "CausalRCA":
        RCA_Model = CausalRCA
    elif args.RCA_Model == "RUN":
        RCA_Model = RUN
    elif args.RCA_Model == "PC":
        RCA_Model = PC
    elif args.RCA_Model == "LiNGAM":
        RCA_Model = LiNGAM
    elif args.RCA_Model == "GES":
        RCA_Model = GES

    # 固定随机种子
    seed_everything(args.random_seed)
    
    # 设置运行环境
    os.environ ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    if torch.cuda.is_available():
        args.device = torch.device("cuda") if args.gpu else torch.device("cpu")
    
    record_file_name = f"res/dn{args.data_name}_rm{args.RCA_Model}_sg{args.Segmentation}_rw{args.Filtering_nodes}_ut{args.useless_metric_th}_ct{args.change_point_th}_ws{args.win_size}_hd{args.hid_dims}_ca{args.contra_a}_pw{args.pred_win_size}_no{args.nocycle_order}_gt{args.graph_edge_th}_ch{args.contrastive_hid_dims}_sl{args.sparse_loss}_cb{args.contrastive_batch_size}_ce{args.contrastive_epochs}_us{args.used_state}_ct{args.contra_theta}_ft{args.filter_theta}_rf{args.reverse_filter}_fs{args.filter_scope}_at{args.contra_adj_theta}_rt{args.pretrain_recon_theta}_ep{args.epochs}_ft{args.filter_th}_ff{args.filter_first}.txt"
    print(args)
    
    # 检查保存点
    if args.check_saved:
        try:
            with open(record_file_name,"r") as f:
                res = json.load(f)
            if res["pr@avg"] > -1:
                print(f"Existing record {record_file_name}")
                print(res)
                exit()
        except Exception as e: 
            pass
    
    # 遍历数据集
    with open(f'dataset/{args.data_path}', 'rb') as f:
        data = pkl.load(f) 
        
    # 遍历全部的故障样本
    start_time = time.time()
    count = 0  
    k = args.evaluate_k
    hits = np.zeros(k+1)
    sb_min_state_win_size_count = 0
    sb_min_state_win_size_list = []
    for case in tqdm(data):
        
        observation_data = case["observation_data"].astype('float32')
        anomaly_data = case["anomaly_data"].astype('float32')
        fault_occur_time = len(observation_data)
        # if fault_occur_time - args.min_state_win_size < 0:
        #     sb_min_state_win_size_count +=1
        #     sb_min_state_win_size_list.append(fault_occur_time)
        #     print(f"sb min_state_win_size. fault_occur_time:{fault_occur_time}")
        #     continue
        observation_data = pd.concat([observation_data,anomaly_data])
        
        count +=1
        if args.debug:
            if count < args.debug_case:
                continue
            
        root_name = case["root_name"]
        args.root_name = root_name
        # plot_metrics(observation_data,f"{args.data_name}_{count}",root_name)
        
        # 状态划分
        useless_metric_name = []
        
        smodel = 0
        if args.Segmentation == "Contrastive":
            smodel = pre_train(observation_data,args)
            smodel.eval()
            
        if args.filter_first:
            if args.Filtering_nodes != "None":
                useless_metric_name = useless_filter(observation_data,smodel,args,fault_occur_time)
                print(f"useless_metric_name:{useless_metric_name}")
            
        if args.Segmentation != "None": # Segmentation 和 Filtering_nodes 的先后顺序是否需要讨论？
            # 每个case预训练一个分割模型
            observation_datav2,fault_occur_time= segmentationv2(observation_data,smodel,args,fault_occur_time)
            observation_data = observation_datav2
        
        if not args.filter_first:
            if args.Filtering_nodes != "None":
                useless_metric_name = useless_filter(observation_data,smodel,args,fault_occur_time)
                print(f"useless_metric_name:{useless_metric_name}")
        
        if args.debug:
            root_name = case["root_name"]
            plot_metrics(observation_data,count,root_name)
        
        # 过滤无波动指标
        observation_data = drop_coliner_data(observation_data)

        # 过滤无关指标/冗余指标
        if args.Filtering_nodes == "Remove_before":
            observation_data = drop_redundancy_data(observation_data,useless_metric_name)
        
        root_name = case["root_name"] # todo done 现在不同数据集的root_name格式不一样，有的是数组（多个根因），有的是字符串（单个根因）
        names = list(observation_data)
        # plot_metrics(observation_data,count,root_name)
        
        if len(names) == 0:
            # for i in range(k):
            #     if names[0] in root_name:
            #         hits[i+1] += 1
            print(f"all metrics are filtered for failure case {count}")
            continue
        
        # 此处需要注意的是，如果上述过滤后，只剩下一个指标，那么需要直接进行评估，而非通过模型计算根因分数
        if len(names) == 1:
            for i in range(k):
                if names[0] in root_name:
                    hits[i+1] += 1
            continue
    
        # 根据不同模型，获取各节点的根因分数
        scores,graph = RCA_Model(observation_data,args,root_name,count,useless_metric_name)
        
        # Filtering_nodes
        if args.Filtering_nodes == "Remove_after":
            graph = drop_redundancy_grahp(graph,useless_metric_name,names,args)
            temp_names = []
            for i in names:
                if i not in useless_metric_name:
                    temp_names.append(i)
            names = temp_names
        elif args.Filtering_nodes == "Reweight":
            graph = reweight_redundancy_grahp(graph,useless_metric_name,names,args)
        elif args.Filtering_nodes == "End2End":
            if args.end2end_after_filter:
                graph = drop_redundancy_grahp(graph,useless_metric_name,names,args)
                temp_names = []
                for i in names:
                    if i not in useless_metric_name:
                        temp_names.append(i)
                names = temp_names
        
        # 再算一遍分数根因分数
        scores, hit_sum = evaluate(graph,names,root_name,args,args.epochs-1,count)
    
        # recording for evaluation
        score_dict = {}
        for i,s in enumerate(scores):
            score_dict[names[i]] = s
        sorted_dict = sorted(score_dict.items(), key=lambda item:item[1], reverse=True)
        rank_names = [i[0] for i in sorted_dict]
        
        print(f"root cause:{root_name}")
        print(f"rank res:{sorted_dict}")
        
        for i in range(k):
            for j in range(i+1):
                if j >= len(rank_names):
                    continue
                if rank_names[j] in root_name:
                    hits[i+1] += 1
        
        if args.debug:
            if count >= args.debug_case:
                break

    end_time = time.time()
    
    # Evaluation
    prs = np.zeros(k+1)
    prs_sum = 0
    res = {}
    for i in range(k):
        prs[i+1] = hits[i+1] / count
        prs_sum += prs[i+1]
        res[f"pr@{i+1}"] = prs[i+1]
    prs_avg = prs_sum / k
    res["pr@avg"] = prs_avg
    res["case_num"] = count
    res["used time"] = end_time-start_time

    # 打印并保存结果
    print(res)
    # print(f"sb_min_state_win_size_count:{sb_min_state_win_size_count} sb_min_state_win_size_list:{sb_min_state_win_size_list}")
    print(f"to_samll_sega:{args.to_samll_sega} last_nochange_count:{args.last_nochange_count} future_nochange_count:{args.future_nochange_count}  to_samll_sep:{args.to_samll_sep}")
    
    with open(record_file_name,"w") as fw:
        json.dump(res,fw)
    print(f"save as {record_file_name}")

main()