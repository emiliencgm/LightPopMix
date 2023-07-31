"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from precalcul import precalculate
import torch
import torch.nn.functional as F
import world
import numpy as np
import math
#=============================================================BPR loss============================================================#
class BPR():
    def __init__(self):
        return 
    def bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        return loss/world.config['batch_size']
#=============================================================BPR + CL loss============================================================#
class InfoNCE_loss():
    def __init__(self):
        self.tau = world.config['temp_tau']

    def infonce_loss(self, batch_user, batch_pos, aug_users1, aug_items1, aug_users2, aug_items2):

        # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/self.config['batch_size']
        
        contrastloss = self.info_nce_loss_overall(aug_users1[batch_user], aug_users2[batch_user], aug_users2) \
                        + self.info_nce_loss_overall(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2)
        return contrastloss
    
    def infonce_loss_batch(self, aug_users1, aug_items1, aug_users2, aug_items2):
        
        contrastloss = self.cal_infonce_loss(aug_users1, aug_users2) + self.cal_infonce_loss(aug_items1, aug_items2)
        return contrastloss

    def info_nce_loss_overall(self, z1, z2, z_all):
        '''
        z1--z2: pos,  z_all: neg\n
        return: InfoNCEloss
        '''
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        positive_pairs = (between_sim)
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))#TODO softplus
        #print('positive_pairs / negative_pairs',max(positive_pairs / negative_pairs))
        loss = loss/world.config['batch_size']
        return loss
    
    def cal_infonce_loss(self, z1, z2):
        '''
        BC implementation
        '''
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        ratings = torch.matmul(z1, torch.transpose(z2, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        return loss


    def sim(self, z1: torch.Tensor, z2: torch.Tensor, mode='inner_product'):#TODO
        '''
        计算一个z1和一个z2两个向量的相似度/或者一个z1和多个z2的各自相似度。
        即两个输入的向量数（行数）可能不同。
        '''
        if mode == 'inner_product':
            if z1.size()[0] == z2.size()[0]:
                #return F.cosine_similarity(z1,z2)
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                return torch.sum(torch.mul(z1,z2) ,dim=1)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())
        elif mode == 'cos':
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1,z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())

#=============================================================Myself Adaptive loss============================================================#
class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()

        #1-->2-->2-->1
        # self.linear1=torch.nn.Linear(in_dim,3*in_dim)
        # self.activation1=torch.nn.ReLU()
        # self.linear2=torch.nn.Linear(3*in_dim, in_dim)
        # self.activation2=torch.nn.ReLU()
        # self.linear3=torch.nn.Linear(in_dim,1)
        #TODO from CV projector:
        self.linear = torch.nn.Linear(in_dim,4*in_dim)
        self.BatchNorm = torch.nn.BatchNorm1d(4*in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.ELU()
        self.linear_hidden = torch.nn.Linear(4*in_dim,4*in_dim)
        self.linear_out = torch.nn.Linear(4*in_dim, 1)

    def forward(self, x):
        #TODO Architecture suboptimal
        # x = self.linear1(x)
        # x = self.activation1(x)
        # x = self.linear2(x)
        # x = self.activation2(x)
        # x = self.linear3(x)
        x = self.linear(x)
        # x = self.BatchNorm(x)
        x = self.activation(x)
        x = self.linear_hidden(x)
        x = self.activation(x)
        x = self.linear_out(x)
        x = torch.sigmoid(x)
        return x


class Adaptive_loss(torch.nn.Module):
    def __init__(self, config, model:LightGCN, precal:precalculate):
        super(Adaptive_loss, self).__init__()
        self.config = config
        self.model = model
        self.precal = precal
        self.tau = config['temp_tau']
        self.f = lambda x: torch.exp(x / self.tau)
        self.MLP_model = MLP(5+2*0).to(world.device)

    def adaptive_loss(self, users_emb, pos_emb, ada_coef):
        '''
        '''
        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        theta = torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))
        M = torch.arccos(torch.clamp(ada_coef,-1+1e-7,1-1e-7))
        ratings_diag = torch.cos(theta + M)
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        loss = torch.mean(torch.negative((torch.log(numerator) -  torch.log(denominator))))

        return loss

    def get_popdegree(self, batch_user, batch_pos_item):
        with torch.no_grad():
            pop_user = torch.tensor(self.precal.popularity.user_pop_degree_label).to(world.device)[batch_user]
            pop_item = torch.tensor(self.precal.popularity.item_pop_degree_label).to(world.device)[batch_pos_item]
        return pop_user, pop_item 
    
    def get_centroid(self, batch_user, batch_pos_item, centroid='eigenvector', aggr='mean', mode='GCA'):
        with torch.no_grad():
            batch_weight = self.precal.centroid.cal_centroid_weights_batch(batch_user, batch_pos_item, centroid=centroid, aggr=aggr, mode=mode)
        return batch_weight
    
    def get_commonNeighbor(self, batch_user, batch_pos_item):
        with torch.no_grad():
            n_users = self.model.num_users
            csr_matrix_CN_simi = self.precal.common_neighbor.CN_simi_mat_sp
            batch_user, batch_pos_item = np.array(batch_user.cpu()), np.array(batch_pos_item.cpu())
            batch_weight1 = csr_matrix_CN_simi[batch_user, batch_pos_item+n_users]
            batch_weight2 = csr_matrix_CN_simi[batch_pos_item+n_users, batch_user]
            batch_weight1 = torch.tensor(np.array(batch_weight1).reshape((-1,))).to(world.device)
            batch_weight2 = torch.tensor(np.array(batch_weight2).reshape((-1,))).to(world.device)
        return batch_weight1, batch_weight2

    def get_mlp_input(self, features):
        '''
        features = [tensor, tensor, ...]
        '''
        U = features[0].unsqueeze(0)
        for i in range(1,len(features)):
            U = torch.cat((U, features[i].unsqueeze(0)), dim=0)
        return U.T


    def get_coef_adaptive(self, batch_user, batch_pos_item, method='mlp', mode='eigenvector'):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])\n
        the bigger, the more reliable, the more important
        '''
        if method == 'mlp':
            batch_weight_pop_user, batch_weight_pop_item = self.get_popdegree(batch_user, batch_pos_item)
            # batch_weight_pop_user = torch.ones_like(batch_weight_pop_user)*math.log(self.precal.popularity.max_pop_u)-torch.log(batch_weight_pop_user)#TODO problem of grandeur and +-
            # batch_weight_pop_item = torch.ones_like(batch_weight_pop_item)*math.log(self.precal.popularity.max_pop_i)-torch.log(batch_weight_pop_item)
            #batch_weight_homophily = self.get_homophily(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = torch.log(batch_weight_pop_user), torch.log(batch_weight_pop_item)
            batch_weight_centroid = self.get_centroid(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')
            batch_weight_centroid = torch.ones_like(batch_weight_centroid) - batch_weight_centroid#TODO 反向centroid
            batch_weight_commonNeighbor1, batch_weight_commonNeighbor2 = self.get_commonNeighbor(batch_user, batch_pos_item)
            features = [batch_weight_pop_user, batch_weight_pop_item, batch_weight_centroid, batch_weight_commonNeighbor1, batch_weight_commonNeighbor2]
            
            # for i in range(self.config['latent_dim_rec']):
            #     features.append(batch_weight_emb_user[:,i])
            # for i in range(self.config['latent_dim_rec']):
            #     features.append(batch_weight_emb_item[:,i])
            
            batch_weight = self.get_mlp_input(features)
            batch_weight = self.MLP_model(batch_weight)

        else:
            batch_weight = None
            raise TypeError('adaptive method not implemented')
        
        self.batch_weight = batch_weight
        return batch_weight