"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import world
import torch
from torch import nn
from dataloader import dataset
from precalcul import precalculate
from torch_geometric.nn import LGConv
from torch_geometric.nn import GCNConv
from torch.nn import ModuleList
import numpy as np

class LGN_Encoder(torch.nn.Module):
    def __init__(self, n_layers, num_users, num_items):
        super(LGN_Encoder, self).__init__()
        self.n_layers = n_layers
        self.num_users, self.num_items = num_users, num_items
        self.convs = ModuleList([LGConv() for _ in range(n_layers)])
        self.alpha = 1. / (n_layers + 1)
        self.bn = torch.nn.BatchNorm1d(self.dim)

    def forward(self, x, edge_index):
        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha
        out = self.bn(out)
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items
    
class GCN_Encoder(torch.nn.Module):
    def __init__(self, n_layers, num_users, num_items):
        super(GCN_Encoder, self).__init__()
        self.n_layers = n_layers
        self.dim = world.config['latent_dim_rec']
        self.num_users, self.num_items = num_users, num_items
        self.convs = ModuleList([GCNConv(self.dim,self.dim) for _ in range(n_layers)])
        self.alpha = 1. / (n_layers + 1)
        self.bn = torch.nn.BatchNorm1d(self.dim)

    def forward(self, x, edge_index):
        # out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            # out = out + x * self.alpha
        x = self.bn(x)#TODO BatchNorm
        users, items = torch.split(x, [self.num_users, self.num_items])
        return users, items


    
class LightGCN(nn.Module):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.precal = precal 
        self.__init_weight()
        self.convs = ModuleList([LGConv() for _ in range(self.n_layers)])
        self.alpha = 1. / (self.n_layers + 1)

        self.projector_user = Projector()#TODO 投影头
        self.projector_item = Projector()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("user:{}, item:{}".format(self.num_users, self.num_items))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['num_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        if world.config['init_method'] == 'Normal':
            world.cprint('use NORMAL distribution UI for Embedding')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        elif world.config['init_method'] == 'Xavier':
            world.cprint('use Xavier_uniform distribution UI for Embedding')
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1.0)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1.0)
        else:
            raise TypeError('init method')
        
        self.f = nn.Sigmoid()

        # if self.dataset.Graph is None:
        #     self.Graph = self.dataset.getSparseGraph()
        # else:
        #     self.Graph = self.dataset.Graph
        
        #TODO num_node = num_users + num_items
        self.edge_index = self.dataset.edge_index
        self.graph = self.dataset.graph_pyg

        print(f"GCL Model is ready to go!")

    def computer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """
        users_emb0 = self.embedding_user.weight
        items_emb0 = self.embedding_item.weight
        x = torch.cat([users_emb0, items_emb0])
        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, self.edge_index)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items
    
    def computer_per_layer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """
        users_emb0 = self.embedding_user.weight
        items_emb0 = self.embedding_item.weight
        x = torch.cat([users_emb0, items_emb0])
        embs_per_layer = []
        embs_per_layer.append(x)
        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, self.edge_index)
            embs_per_layer.append(x)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items, embs_per_layer

    def view_computer(self, x, edge_index, edge_weight=None):
        try:
            x, edge_index, edge_weight = x.to(world.device), edge_index.to(world.device), edge_weight.to(world.device)
        except:
            x, edge_index = x.to(world.device), edge_index.to(world.device)

        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items
    
    #================Pop=================#
    def getItemRating(self):
        '''
        获取输入items1, items2对全部user的平均得分
        return: rating1=Hot, rating2=Cold
        '''
        itemsPopDict = self.precal.popularity.ItemPopGroupDict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        #rating = self.f(torch.matmul(items_emb, users_emb.t()))#TODO 内积后过Sigmoid()作为输出Rating
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#
    
    def getUsersRating(self, users):
        '''
        先执行一次model.computer().
        return rating=指定users对每个item做内积后过Sigmoid()
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))#TODO 为避免不同样本预测分值的数量级差异导致的梯度数值差异，BPRloss计算时通常用Sigmoid将正负样本预测分插值映射至01之间。
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        '''
        先执行一次model.computer().
        return: users, pos_items, neg_items各自的初始embedding(item在聚合KG信息前的embedding)和LightGCN更新后的embedding
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        All_embs = [all_users, all_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, All_embs
    
    def bpr_loss(self, users, pos, neg):
        '''
        输入一个batch的users、pos_items、neg_items
        reG_loss = users、pos_items、neg_items初始embedding的L2正则化loss
        reC_loss = Σ{ softplus[ (ui,negi) - (ui,posi) ] }
        '''
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))#/float(len(users))
        #TODO 这里的reg数量级有问题？除以了batch？？
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        return loss, reg_loss
    

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, out_dim, precal:precalculate):
        super(Classifier, self).__init__()
        self.input_dim = input_dim

        self.all_label = precal.popularity.item_pop_group_label
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4), 
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, out_dim),
            torch.nn.Softmax(dim=-1)
        ).to(world.device)

        self.criterion = nn.CrossEntropyLoss()

    # def forward(self, x):
    #     return self.net(x)
    
    def cal_loss_and_test(self, inputs, batch_item):
        '''
        return loss and test accuracy of the same batch before update
        '''
        batch_item = batch_item.cpu()
        batch_label = torch.tensor(self.all_label[batch_item]).to(world.device)
        outputs = self.net(inputs)
        CE_loss = self.criterion(outputs, batch_label)

        predicted_labels = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((predicted_labels == batch_label).float())

        return CE_loss, accuracy
    
    # def test(self, inputs, batch_item):
    #     true_labels = self.all_label[batch_item]
    #     model_output = self.net(inputs)
    #     predicted_labels = torch.argmax(model_output, dim=1)
    #     accuracy = torch.mean((predicted_labels == true_labels).float())

    #     return accuracy



class Projector(torch.nn.Module):
    def __init__(self, output_dim=world.config['latent_dim_rec'], input_dim=world.config['latent_dim_rec']):
        super(Projector, self).__init__()        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4), 
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, output_dim)
        ).to(world.device)

    def forward(self, embs):
        return self.net(embs)