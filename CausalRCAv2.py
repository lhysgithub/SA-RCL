import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from utilsv2 import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class CModel(nn.Module):
    def __init__(self,args,dims):
        super(CModel, self).__init__()
        self.sep_encoder = nn.Linear(args.win_size,args.hid_dims)
        self.sep_decoder = nn.Linear(args.hid_dims,args.pred_win_size)
        self.joint_encoder = nn.Linear(dims,args.hid_dims)
        self.joint_decoder = nn.Linear(args.hid_dims,dims)
        self.args = args
        
    def forward(self, x, granger_model, pre_train=True):
        if pre_train:
            x0 = x[:,:self.args.win_size,:]
            y0 = x[:,self.args.win_size:,:]
        else:
            x0 = x
        # Z = (I-A^T)X
        # B = (I-A^T)
        x1 = self.sep_encoder(x0.permute(0,2,1)).permute(0,2,1) # bwd->bhd # 
        x2 = granger_model(x1) # bhd->bhd
        x3 = self.joint_encoder(x2) # bhd -> bhh
        x4 = self.joint_decoder(x3) # bhh -> bhd
        x5 = self.sep_decoder(x4.permute(0,2,1)).permute(0,2,1) # bhd->bpd
        
        if pre_train:
        # recon_loss = F.mse_loss(x5,x0)
            pred_loss = F.mse_loss(x5,y0)
        else:
            pred_loss = 0
    
        return x1, x3, x5, granger_model.adj_A, pred_loss
   
class Granger(nn.Module):
    def __init__(self,args,adj1):
        super(Granger, self).__init__()
        self.adj_A = nn.Parameter(torch.autograd.Variable(adj1, requires_grad=True)) # bdd
        
    def forward(self, x):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')
        
        A = F.relu(self.adj_A) # bdd
        # A = A - torch.diag(torch.diag(A)) # dd 
        B = (torch.eye(A.shape[-1]).to(x.device).unsqueeze(0).repeat(A.shape[0],1,1) - (A.permute(0,2,1))) # bdd
        x2 = torch.einsum("bde,bwd->bwe",B, x) # bhd->bhd
        return x2
    
class Granger2(nn.Module):
    def __init__(self,args,adj1):
        super(Granger2, self).__init__()
        self.adj_A = nn.Parameter(torch.autograd.Variable(adj1, requires_grad=True)) # dd
        
    def forward(self, x):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')
        
        A = F.relu(self.adj_A) # dd
        A = A - torch.diag(torch.diag(A)) # dd 
        B = (torch.eye(A.shape[-1]).to(x.device) - (A.permute(1,0))) # dd
        x2 = torch.einsum("de,bwd->bwe",B, x) # bhd->bhd
        return x2
         


def train(model, train_data, args, optimizer, useless_metric_index,epoch,modelg,optimizerg):
    pred_loss_list = []
    loss_list = []
    nocycle_loss_list = []
    sparse_loss_list = []
    related_loss_list = []

    for x,y in train_data:
        b,w,d = x.shape
        x = x.to(args.device)
        target = y.to(args.device)

        optimizer.zero_grad()
        optimizerg.zero_grad()
        
        _, _, output, origin_A, _ = model(x,modelg,False)
        origin_A = F.relu(origin_A)

        if torch.sum(output != output):
            print('nan error\n')

        # reconstruction accuracy loss 
        # pred_loss = 1e2*F.mse_loss(output, target)
        invers_useless_metric_index = torch.ones_like(useless_metric_index) - useless_metric_index
        if args.Filtering_nodes == "End2End":
            pred_loss = (1e2*invers_useless_metric_index*((output-target)**2).mean(0).mean(0)).mean() + args.filter_th*(1e2*useless_metric_index*((output-target)**2).mean(0).mean(0)).mean()
        else:
            pred_loss = 1e2*((output-target)**2).mean()
        
        # sparse_loss: 边数量正则
        sparse_loss = args.sparse_loss*torch.sum(torch.abs(origin_A)) 
        
        # related_loss: 相关性loss正则
        related_loss = args.filter_theta*(torch.abs(origin_A).sum(0)*useless_metric_index + torch.abs(origin_A).sum(1)*useless_metric_index).sum()

        # compute nocycle_loss
        nocycle_loss = 1*h_A(origin_A, args.nocycle_order) # bug，明明有环，但loss为0  # len(origin_A) 太大有问题
        
        loss = pred_loss + nocycle_loss + sparse_loss
        
        if args.Filtering_nodes == "End2End":
            loss += related_loss
        
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizerg.step()

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
       
        loss_list.append(loss.item())
        pred_loss_list.append(pred_loss.item())
        nocycle_loss_list.append(nocycle_loss.item())
        sparse_loss_list.append(sparse_loss.item())
        related_loss_list.append(related_loss.item())
    
    if args.debug:
        print(f"epoch:{epoch} all_loss:{np.mean(loss_list)} pred_loss:{np.mean(pred_loss_list)} nocycle_loss:{np.mean(nocycle_loss_list)} sparse_loss:{np.mean(sparse_loss_list)} related_loss_list:{np.mean(related_loss_list)}")

    return graph, origin_A


def CausalRCA(data, args, root_name,case_number,useless_metric_name):
    names = list(data)
    train_data = PreDataset(data.to_numpy(),args)
    train_data_loader = DataLoader(train_data,batch_size=len(train_data),shuffle=True)
    adj = torch.ones((len(names),len(names))).to(args.device)*0.5
    model = CModel(args,len(names)).to(args.device).train()
    # model = args.smodel
    modelg = Granger2(args, adj).to(args.device).train()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    optimizerg = optim.Adam(modelg.parameters(),lr=args.lr)
    best_hit_sum = -1
    # 获取无关指标下标
    useless_metric_index=get_index(names,useless_metric_name,args)
    
    scores = None
    for epoch in range(args.epochs):
        graph, origin_A = train(model,train_data_loader,args,optimizer,useless_metric_index,epoch,modelg,optimizerg)
    
        if args.debug:
            scores, hit_sum = evaluate(graph,names,root_name,args,epoch,case_number) 
            if best_hit_sum < hit_sum:
                best_hit_sum = hit_sum
                best_score = scores
                print(f"find better hit_sum {hit_sum} at epoch {epoch}")
                # print("find best")
    # print(f"better hit_sum {best_hit_sum}")
        
    return scores,graph # best_score 可能会导致标签泄露
    
