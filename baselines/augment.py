"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
import numpy as np
from utils import randint_choice
import scipy.sparse as sp
import world
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from precalcul import precalculate
import time
import faiss
import torch_sparse
from scipy.sparse import csr_matrix
from dataloader import dataset
import torch_geometric
import copy


class ED_Uniform():
    def __init__(self, config, model:LightGCN, precal:precalculate):
        self.config = config
        self.model = model
        self.precal = precal
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def Edge_drop_random(self, p_drop):
        '''
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.model.num_users + self.model.num_items
        #注意数组复制问题！
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_random(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_random(p_drop)

class ED_Adaptive():
    def __init__(self, config, model:LightGCN, precal:precalculate):
        self.config = config
        self.model = model
        self.precal = precal
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

        return edge_index[:, sel_mask]
    def Edge_drop_weighted(self, p_drop):
        '''
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.model.num_users + self.model.num_items
        #注意数组复制问题！
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_weighted(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_weighted(p_drop)

class RW_Uniform(ED_Uniform):
    def __init__(self, config, model, precal):
        super(RW_Uniform, self).__init__(config, model, precal)

    def Random_Walk(self, p_drop):
        aug_g = []
        for layer in range(self.config['num_layers']):
            aug_g.append(self.Edge_drop_random(p_drop))
        return aug_g

    # def computer(self, p_drop):
    #     aug_g = self.Random_Walk(p_drop)
    #     return self.model.view_computer(aug_g)

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Random_Walk(p_drop)
        self.augAdjMatrix2 =  self.Random_Walk(p_drop)


class SVD_Augment():
    def __init__(self, config, model:LightGCN, precal:precalculate):
        self.config = config
        self.model = model
        self.precal = precal
        self.n_layers = config['num_layers']
        self.num_users, self.num_items = self.model.num_users, self.model.num_items

    def reconstruct_graph_computer_origin(self):
        users_emb = self.model.embedding_user.weight
        items_emb = self.model.embedding_item.weight
        embs_u = [users_emb]    
        embs_i = [items_emb]    
        for layer in range(self.n_layers):
            #propagation for user, neighbors are items
            vt_ei = self.precal.svd.svd_v_T @ embs_i[layer]
            emb_u = self.precal.svd.u_mul_s @ vt_ei
            embs_u.append(emb_u)
            #propagation for item, neighbors are users
            ut_eu = self.precal.svd.svd_u_T @ embs_u[layer]
            emb_i = self.precal.svd.v_mul_s @ ut_eu
            embs_i.append(emb_i)

        embs_u = torch.stack(embs_u, dim=1)
        light_out_user = torch.mean(embs_u, dim=1)
        embs_i = torch.stack(embs_i, dim=1)
        light_out_item = torch.mean(embs_i, dim=1)


        return light_out_user, light_out_item

    def reconstruct_graph_computer(self):
        with torch.no_grad():
            users_emb = self.model.embedding_user.weight
            items_emb = self.model.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            graph = self.precal.svd.u_mul_s_mul_v_T
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(graph, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items
    
    