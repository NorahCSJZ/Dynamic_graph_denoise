# encoding: utf-8
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] = "2,3"
# import tensorflow as tf
import sys
import math

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import LocallyLinearEmbedding
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy import sparse
from einops import rearrange


class Spatial_Attention_0(nn.Module):
    def __init__(self):
        super(Spatial_Attention_0, self).__init__()
        self.V1_h0 = nn.Parameter(torch.empty(size=(L10, L11), dtype=torch.float64))
        nn.init.trunc_normal_(self.V1_h0.data, mean=-0.1, std=0.1)
        self.w1_h0 = nn.Parameter(torch.empty(size=2*L10, dtype=torch.float64))
        nn.init.trunc_normal_(self.w1_h0.data, mean=-0.1, std=0.1)

        self.softmax = nn.Softmax(dim=0)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # f_tar_samp: (2*dim_hop0)*[0]*6400
        # V1_h0: L10*L11
        # w1_h0: (2*L10)
        # dim_hop0 = L11

        temp_sp1 = torch.einsum('ij,jkl->ikl', self.V1_h0, x[0:dim_hop0, :, :])  # L10*[0]*6400 --one layer of network
        temp_sp2 = torch.einsum('ij,jkl->ikl', self.V1_h0,
                             x[dim_hop0:2 * dim_hop0, :, :])  # L10*[0]*6400 --one layer of network
        B = self.softmax(self.leaky_relu(torch.einsum('j,jkl->kl', self.w1_h0, torch.concat([temp_sp1, temp_sp2], 0))))  # [0]*6400

        return torch.permute(B, (1, 0))  # 6400*[0]


class Spatial_Attention_1(nn.Module):
    def __init__(self):
        super(Spatial_Attention_1, self).__init__()
        self.V1_h1 = nn.Parameter(torch.empty(size=(L11, n_dim), dtype=torch.float64))
        nn.init.trunc_normal_(self.V1_h1.data, mean=-0.1, std=0.1)

        self.w1_h1 = nn.Parameter(torch.empty(2*L11, dtype=torch.float64))
        nn.init.trunc_normal_(self.w1_h1.data, mean=-0.1, std=0.1)

        self.softmax = nn.Softmax(dim=0)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        temp_sp1 = torch.einsum('ij,jklm->iklm', self.V1_h1, torch.permute(x[:, :, :, 0:n_dim], (3, 2, 1, 0)))  # L11*([1]/[0])*[0]*6400 --one layer of network
        temp_sp2 = torch.einsum('ij,jklm->iklm', self.V1_h1, torch.permute(x[:, :, :, n_dim:2 * n_dim], (3, 2, 1, 0)))  # L11*([1]/[0])*[0]*6400 -- one layer of network
        temp_sp3 = torch.einsum('j,jklm->klm', self.w1_h1, torch.concat([temp_sp1, temp_sp2], 0))
        B = self.softmax(self.leaky_relu(temp_sp3))  # ([1]/[0])*[0]*6400

        return torch.permute(B, (2, 1, 0)) #6400*[0]*([1]/[0]); 6400*1*[0]


class Aggregate_Att_Mean(nn.Module):
    def __init__(self, samples,  sup_sizes):
        super(Aggregate_Att_Mean, self).__init__()
        self.spatial_att_mean_1 = Spatial_Attention_1()
        self.spatial_att_mean_0 = Spatial_Attention_0()

        self.V1_h1 = nn.Parameter(torch.empty(size=(L11, n_dim), dtype=torch.float64))
        nn.init.trunc_normal_(self.V1_h1.data, mean=-0.1, std=0.1)
        self.weights_hops_1 = nn.Parameter(torch.empty(size=(dim_hop0, dim_hop1), dtype=torch.float64))
        nn.init.trunc_normal_(self.weights_hops_1.data, mean=-0.1, std=0.1)

        self.sigmoid = nn.Sigmoid()


        self.samples = samples
        self.sup_sizes = sup_sizes

    def forward(self, x):
        sup_sizes = [1] + self.sup_sizes  # [1, [0], [1]]
        ## k=1: update node representations of Hop-0
        dim_temp = np.int32(sup_sizes[2] / sup_sizes[1])
        # gather neighbors at hop-1
        samples_temp = torch.reshape(self.samples[:, -sup_sizes[2]:],
                                     [-1, sup_sizes[1], dim_temp])  # shape: 6400*[0]*([1]/[0])
        fea_mat_samples_temp = torch.select(x,
                                            samples_temp)  # shape: 6400*[0]*([1]/[0])*42 -- torch.nn.Embeddings

        # gather target nodes for hop-1 and replicate attributes
        samples_temp_t = self.samples[:, -(sup_sizes[1] + sup_sizes[2]):-sup_sizes[2]]  # shape: 6400*[0]
        target_temp = torch.unsqueeze(samples_temp_t, 2)  # 6400*[0]*1
        target_temp = torch.tile(target_temp, [1, 1, dim_temp])  # 6400*[0]*([1]/[0])
        fea_mat_samples_temp_t = torch.select(x,
                                              target_temp)  # shape: 6400*[0]*([1]/[0])*42  -- torch.nn.Embeddings

        # learn attention for hop-1
        fea_tar_samp = torch.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 3)  # shape: 6400*[0]*([1]/[0])*84

        Beta_hop1 = self.spatial_att_mean_1(fea_tar_samp)

        temp_aam_0 = torch.einsum('ij,jklm->iklm', self.V1_h1,
                               torch.transpose(fea_mat_samples_temp,(3, 2, 1, 0)))  # shape: L11*([1]/[0])*[0]*6400
        temp_aam_1 = torch.matmul(torch.unsqueeze(Beta_hop1, 2),
                               torch.permute(temp_aam_0, (3, 2, 1, 0)))  # shape: 6400*[0]*1*L11 ???
        fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=2)  # shape: 6400*[0]*L11

        # concatenate and transform
        fea_mat_samples_temp_t_emb = torch.einsum('ij,jkl->ikl', self.V1_h1, torch.permute(fea_mat_samples_temp_t[:, :, 0, :],
                                                                                           (2, 1, 0)))  # L11*[0]*6400
        con_hop1 = torch.concat([fea_mat_samples_temp_t_emb, torch.permute(fea_hop_mean_att, (2, 1, 0))],
                             0)  # shape:(L11*2)*[0]*6400, dim_hop1=L11*2
        hop1 = self.sigmoid(torch.einsum('ij,jkl->ikl', self.weights_hops_1,
                                    con_hop1))  # shape: (dim_hop0,dim_hop1)*((L11*2),[0]*6400)=dim_hop0*[0]*6400


        ## k=1: update node representations of nodes in batch
        dim_temp = np.int32(sup_sizes[1])
        # gather neighbors at hop-0
        samples_temp = self.samples[:, -(sup_sizes[1] + sup_sizes[2]):-sup_sizes[2]]  # shape: 6400*[0]
        fea_mat_samples_temp = torch.select(x, samples_temp)  # shape: 6400*[0]*42
        # gather target nodes for hop-0 and replicate attributes
        samples_temp_t = self.samples[:, 0]  # shape: 6400*1
        target_temp = torch.tile(torch.unsqueeze(samples_temp_t, 1), [1, dim_temp])  # 6400*[0]
        fea_mat_samples_temp_t = torch.select(x, target_temp)  # shape: 6400*[0]*42
        # learn attention for hop-0
        fea_tar_samp = torch.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 2)  # shape: 6400*[0]*84

        Beta_hop0 = self.spatial_attention_1(torch.unsqueeze(fea_tar_samp, 1))  # shape: 6400*1*[0]
        # calculate neighbor representation
        #   V1_h1: L11*n_dim
        #   fea_mat_samples_temp: 6400*[0]*42
        temp_aam_0 = torch.einsum('ij,jkl->ikl', self.V1_h1,
                               torch.permute(fea_mat_samples_temp, (2, 1, 0)))  # shape: L11*[0]*6400
        temp_aam_1 = torch.matmul(Beta_hop0, torch.permute(temp_aam_0, (2, 1, 0)))  # shape: 6400*1*L11 ???
        fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=1)  # shape: 6400*L11

        # concatenate and transform
        fea_mat_samples_temp_t_emb = torch.einsum('ij,jk->ik', self.V1_h1,
                                               torch.transpose(fea_mat_samples_temp_t[:, 0, :], (1, 0)))  # L11*6400
        con_hop1 = torch.concat([fea_mat_samples_temp_t_emb, torch.transpose(fea_hop_mean_att, (1, 0))],
                             0)  # shape: (L11*2)*6400, dim_hop1=L11*2
        hop0 = self.sigmoid(torch.einsum('ij,jk->ik', self.weights_hops_1,
                                    con_hop1))  # shape: (dim_hop0,dim_hop1)*((L11*2)*6400)=dim_hop0*6400

        ## k=2: generate neighbor representation
        dim_temp = np.int32(sup_sizes[1])
        # gather target nodes for hop-0 and replicate attributes
        target_temp = torch.unsqueeze(hop0, 1)  # dim_hop0*1*6400
        target_temp = torch.tile(target_temp, [1, dim_temp, 1])  # dim_hop0*[0]*6400
        # learn attention
        fea_tar_samp = torch.concat([target_temp, hop1], 0)  # shape: (2*dim_hop0)*[0]*6400
        Beta_hop = self.spatial_attention_0(fea_tar_samp)  # shape: 6400*[0]

        # calculate neighbor representation
        #   V1_h0: L10*dim_hop0
        #   hop1: dim_hop0*[0]*6400
        temp_aam_0 = torch.einsum('ij,jkl->ikl', self.V1_h0, hop1)  # shape: L10*[0]*6400
        temp_aam_1 = torch.matmul(torch.unsqueeze(Beta_hop, 1),
                               torch.permute(temp_aam_0, (2, 1, 0)))  # shape: 6400*1*L10 ???
        fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=1)  # shape: 6400*L10

        ## concatenate spatial attention
        # Beta_hop1: 6400*[0]*([1]/[0])
        # Beta_hop: 6400*[0]
        Beta_step = torch.concat([torch.unsqueeze(Beta_hop, 2), Beta_hop1], 2)  # 6400*[0]*(1+[1]/[0])

        return torch.transpose(fea_hop_mean_att), Beta_step  # L10*6400 = 42*6400; 6400*[0]*(1+[1]/[0])


class GRU_neighbor(nn.Module):
    def __init__(self, Fea, saps_idx, sup_sizes, n_h_units):
        super(GRU_neighbor, self).__init__()
        self.Wz = nn.Parameter(torch.empty(size=(M, M+n_dim*2), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wz.data, mean=-0.1, std=0.1)

        self.Wr1 = nn.Parameter(torch.empty(size=(M, M+n_dim*2), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wr1.data, mean=-0.1, std=0.1)

        self.Wr2_1 = nn.Parameter(torch.empty(size=(n_dim, M), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wr2_1.data, mean=-0.1, std=0.1)

        self.Wr2_2 = nn.Parameter(torch.empty(size=(n_dim, n_dim*2), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wr2_2.data, mean=-0.1, std=0.1)

        self.Wh_1 = nn.Parameter(torch.empty(size=(M, M), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wh_1.data, mean=-0.1, std=0.1)

        self.Wh_2 = nn.Parameter(torch.empty(size=(M, n_dim*2), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wh_2.data, mean=-0.1, std=0.1)

        self.bz = nn.Parameter(torch.zeros(M, dtype=torch.float64))
        nn.init.trunc_normal_(self.bz.data)
        self.br1 = nn.Parameter(torch.zeros(M, dtype=torch.float64))
        nn.init.trunc_normal_(self.br1.data)
        self.br2 = nn.Parameter(torch.zeros(n_dim, dtype=torch.float64))
        nn.init.trunc_normal_(self.br2.data)
        self.bh = nn.Parameter(torch.zeros(n_dim, dtype=torch.float64))
        nn.init.trunc_normal_(self.bh.data)

        self.saps_idx = saps_idx
        self.sup_sizes = sup_sizes
        self.features = Fea
        self.n_h_units = n_h_units
        self.aggregate_att_mean = Aggregate_Att_Mean(self.saps_idx[i, :, :], self.sup_sizes[i])

    def forward(self, x):
        T = self.features.shape[1]  # T = 50
        n_dim = self.features.shape[2]  # n_dim = 42
        x = x.type(torch.int32)
        X = torch.select(self.features, x)
        for i in range(T):
            xt_temp = X[:, i, :]  # 6400*42
            xt = torch.transpose(xt_temp)  # 42*6400

            # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
            xnt, Beta_step = self.aggregate_att_mean(self.features[:, i, :])  # shape: 42*6400; 6400*[0]*(1+[1]/[0])
            if i == 0:
                rt2_temp = torch.matmul(self.Wr2_2, torch.concat([xt, xnt], 0))  # (42*84)*((42+42)*6400)=42*6400
                rt2 = torch.transpose(F.sigmoid(torch.transpose(rt2_temp) + self.br2))
                ht_temp = torch.matmul(self.Wh_2, torch.concat([xt, torch.multiply(rt2, xnt)], 0))  # (20*84)*(84*6400)=20*6400
                ht_tilde = torch.transpose(F.tanh(torch.transpose(ht_temp) + self.bh))  # 6400*20 + 20 -> 20*6400
                ht = ht_tilde
                hp = ht  # 20*6400
                h_list_tensor = torch.unsqueeze(hp, 1)  # 20*1*6400

                Beta = torch.unsqueeze(Beta_step, 0)  # 1*6400*[0]*(1+[1]/[0])

            else:
                Wr2 = torch.cat([self.Wr2_1, self.Wr2_2], 1)
                Wh = torch.concat([self.Wh_1, self.Wh_2], 1)  # [20*20,20*84] -> 20*104
                Wzr = torch.concat([self.Wz, self.Wr1, Wr2], 0)  # (20+20+42)*104
                bzr = torch.concat([self.bz, self.br1, self.br2], 0)  # (20+20+42)

                zr_temp = torch.matmul(Wzr, torch.concat([hp, xt, xnt], 0))  # (82*104)*((20+42+42)*6400)=82*6400
                zr = torch.transpose(F.sigmoid(torch.transpose(zr_temp) + bzr))  # (20+20+42)*6400: zt, rt1, rt2

                rt1 = zr[self.n_h_units:self.n_h_units * 2, :]  # 20*6400
                rt2 = zr[self.n_h_units * 2:self.n_h_units * 2 + n_dim, :]  # 42*6400
                ht_temp = torch.matmul(Wh, torch.concat([torch.multiply(rt1, hp), xt, torch.multiply(rt2, xnt)], 0))  # 20*6400
                ht_tilde = torch.transpose(torch.tanh(torch.transpose(ht_temp) + self.bh))  # 6400*20 + 20 -> 20*6400

                zt = zr[0:self.n_h_units, :]  # 20*6400
                zt_neg = 1.0 - zt
                ht = torch.multiply(zt_neg, hp) + torch.multiply(zt, ht_tilde)

                hp = ht  # 20*6400
                h_list_tensor = torch.concat([h_list_tensor, torch.unsqueeze(hp, 1)], 1)  # 20*T*6400

                Beta = torch.concat([Beta, torch.unsqueeze(Beta_step, 0)], 0)  # (T)*6400*[0]*(1+[1]/[0])

        h_list_tensor = torch.permute(h_list_tensor, (2, 1, 0))  # 6400*T*20 (M=20)

        return h_list_tensor, Beta


class GRU_Topology(nn.Module):
    def __init__(self, Topo, n_h_units):
        super(GRU_Topology, self).__init__()
        self.Wz_tp = nn.Parameter(torch.empty(size=(M, M+dim_redu), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wz_tp.data, std=0.1)

        self.Wr1_tp = nn.Parameter(torch.empty(size=(M, M+dim_redu), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wr1_tp.data, std=0.1)

        self.Wh_1_tp = nn.Parameter(torch.empty(size=(M, M), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wh_1_tp.data, std=0.1)

        self.Wh_2_tp = nn.Parameter(torch.empty(size=(M, dim_redu), dtype=torch.float64))
        nn.init.trunc_normal_(self.Wh_2_tp.data, mean=-0.1, std=0.1)

        self.bz_tp = nn.Parameter(torch.zeros(M, dtype=torch.float64))
        nn.init.trunc_normal_(self.bz_tp.data, std=0.1)
        self.br1_tp = nn.Parameter(torch.ones(M, dtype=torch.float64))
        nn.init.trunc_normal_(self.br1_tp.data, std=0.1)
        self.bh_tp = nn.Parameter(torch.ones(M, dtype=torch.float64))
        nn.init.trunc_normal_(self.bh_tp.data, std=0.1)


        self.topo = Topo
        self.n_h_units = n_h_units

    def forward(self, x):
        T = self.topo.shape[1]  # T = 50
        n_dim = self.topo.shape[2]  # n_dim = 42
        x = x.type(torch.int32)
        X = torch.select(self.topo, x)  # shape: 6400*50*42

        for i in range(T):
            # print("index i in LSTM:", i)
            xt_temp = X[:, i, :]  # 6400*42
            xt = torch.transpose(xt_temp)  # 42*6400

            if i == 0:
                # print(Wh_2_tp)
                # print(xt)
                ht_temp = torch.matmul(self.Wh_2_tp, xt)  # (20*42)*(42*6400)=20*6400
                ht_tilde = torch.transpose(F.tanh(torch.transpose(ht_temp) + self.bh_tp))  # 6400*20 + 20 -> 20*6400
                ht = ht_tilde

                hp = ht  # 20*6400
                h_list_tensor = torch.unsqueeze(hp, 1)  # 20*1*6400

            else:
                Wh = torch.concat([self.Wh_1_tp, self.Wh_2_tp], 1)  # [20*20,20*42] -> 20*62

                Wzr = torch.concat([self.Wz_tp, self.Wr1_tp], 0)  # (20+20)*62
                bzr = torch.concat([self.bz_tp, self.br1_tp], 0)  # (20+20)

                zr_temp = torch.matmul(Wzr, torch.concat([hp, xt], 0))  # (40*62)*((20+42)*6400)=40*6400
                zr = torch.transpose(F.sigmoid(torch.transpose(zr_temp) + bzr))  # (20+20)*6400: zt, rt1

                rt1 = zr[self.n_h_units:self.n_h_units * 2, :]  # 20*6400
                ht_temp = torch.matmul(Wh, torch.concat([torch.multiply(rt1, hp), xt], 0))  # 20*6400
                ht_tilde = torch.transpose(F.tanh(torch.transpose(ht_temp) + self.bh_tp))  # 6400*20 + 20 -> 20*6400

                zt = zr[0:self.n_h_units, :]  # 20*6400
                zt_neg = 1.0 - zt
                ht = torch.multiply(zt_neg, hp) + torch.multiply(zt, ht_tilde)

                hp = ht  # 20*6400
                h_list_tensor = torch.concat([h_list_tensor, torch.unsqueeze(hp, 1)], 1)  # 20*T*6400

        h_list_tensor = torch.permute(h_list_tensor, (2, 1, 0))  # 6400*T*20 (M=20)

        return h_list_tensor


class Temporal_Attention(nn.Module):
    def __init__(self):
        super(Temporal_Attention, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(r, L), dtype=torch.float64))
        nn.init.trunc_normal_(self.W.data, mean=-0.1, std=0.1)
        self.V = nn.Parameter(torch.zeros(size=(L, 2*M), dtype=torch.float64))
        nn.init.trunc_normal_(self.V.data, mean=-0.1, std=0.1)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        # x: h_list_tensor
        x = torch.einsum('ij,jkl->ikl', self.V, torch.transpose(x, (2, 1, 0)))
        x = self.tanh(x)
        x = torch.einsum('ij,jkl->ikl', self.W, x)
        x = self.softmax(x)

        return torch.permute(x, (2, 0, 1))


class Topo_Attention(nn.Module):
    def __init__(self):
        super(Topo_Attention, self).__init__()
        self.w_tp = nn.Parameter(torch.empty(size=(1, L_tp), dtype=torch.float64))
        nn.init.trunc_normal_(self.w_tp.data, mean=-0.1, std=0.1)
        self.v_tp = nn.Parameter(torch.empty(size=(L_tp, M), dtype=torch.float64))
        nn.init.trunc_normal_(self.v_tp.data, mean=-0.1, std=0.1)

    def forward(self, x):
        # h_1, h_2: 6400*T*M
        # V_tp: L_tp*M
        # w_tp: 1*L_tp
        h_1, h_2 = x
        T = h_1.shape[1]
        for i in range(T):
            h_comb = torch.concat([torch.unsqueeze(h_1[:, i, :], 1), torch.unsqueeze(h_2[:, i, :], 1)], 1)  # 6400*2*M

            temp = F.tanh(torch.einsum('ij,jkl->ikl', self.V_tp, torch.permute(h_comb, (2, 1, 0))))  # L_tp*2*6400
            if i == 0:
                A = F.softmax(torch.einsum('ij,jkl->ikl', self.w_tp, temp), dim=1)  # 1*2*6400
            else:
                A = torch.concat([A, F.softmax(torch.einsum('ij,jkl->ikl', self.w_tp, temp), dim=1)], 0)  # T*2*6400

        return torch.permute(A, (2, 1, 0))  # 6400*2*T


class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(r*2*M, num_classes), dtype=torch.float64))
        nn.init.trunc_normal_(self.W.data, mean=-0.1, std=0.1)
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float64))
        nn.init.trunc_normal_(self.b.data, mean=-0.1, std=0.1)

    def forward(self, x):
        out = torch.matmul(x, self.W) + self.b
        return out



class Model(nn.Module):
    def __init__(self, Fea, Topo,
                                n_hidden_units):
        super(Model, self).__init__()
        self.Fea = Fea
        self.Topo = Topo
        self.n_hidden_units = n_hidden_units
        self.topo_attention = Topo_Attention()  # topo_att_indi_attention(h_list_tensor, h_list_tensor_topo)
        self.mlp1 = MLP_1()
        self.tem_attention = Temporal_Attention()

    def add_parameter(self,samples_idx_batch, support_sizes_batch):
        self.samples_idx_batch = samples_idx_batch
        self.support_sizes_batch = support_sizes_batch
        self.gru_neighbor = GRU_neighbor(self.Fea, self.samples_idx_batch, self.support_sizes_batch,
                                         self.n_hidden_units)
        self.gru_topo = GRU_Topology(self.Topo,
                                     self.n_hidden_units)

    def forward(self, x):
        h_list_tensor, Beta = self.gru_neighbor(x)
        h_list_tensor_topo = self.gru_topo(x)
        Gamma = self.topo_attention(h_list_tensor, h_list_tensor_topo)
        h_1_tmp = torch.multiply(torch.permute(torch.unsqueeze(Gamma[:, 0, :], 1), (0, 2, 1)),
                                 h_list_tensor)  # 6400*T*1 * 6400*T*M = 6400*T*M
        h_2_tmp = torch.multiply(torch.permute(torch.unsqueeze(Gamma[:, 1, :], 1), (0, 2, 1)),
                                 h_list_tensor_topo)  # 6400*T*1 * 6400*T*M = 6400*T*M

        h_att_topo = torch.concat([h_1_tmp, h_2_tmp], 2)  # 6400*T*2M

        # apply temporal attention
        Alpha = self.tem_attention(h_att_topo)  # 6400*r*T

        temp = torch.matmul(Alpha, h_att_topo)  # 6400*r*2M
        shape = temp.shape[1] * temp.shape[2]
        temp1 = torch.reshape(temp, [-1, shape.type(torch.int32)])
        out = self.mlp1(temp1)
        return out, h_list_tensor, Alpha, temp, Beta, Gamma, h_list_tensor_topo


class MLP_3(nn.Module):
    def __init__(self, in_features):
        super(MLP_3, self).__init__()
        self.W1 = nn.Parameter(torch.empty(size=(in_features, 32), dtype=torch.float64))
        nn.init.trunc_normal_(self.W1.data, mean=-0.1, std=0.1)
        self.relu = nn.ReLU()
        self.W2 = nn.Parameter(torch.empty(size=(32, 1), dtype=torch.float64))
        nn.init.trunc_normal_(self.W2.data, mean=-0.1, std=0.1)
        self.b1 = nn.Parameter(torch.zeros(32, dtype=torch.float64))
        nn.init.trunc_normal_(self.b1.data, mean=-0.1, std=0.1)
        self.b2 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        nn.init.trunc_normal_(self.b2.data, mean=-0.1, std=0.1)

    def forward(self, x):
        x = torch.matmul(x, self.W1) + self.b1
        x = self.relu(x)
        x = torch.matmul(x, self.W2) + self.b2
        return x







class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data, self.label





def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2


def normalize(v):
    return F.normalize(v, dim=1, p=2, eps=1e-12)



def minmax_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    tensor_norm = np.ones([n_node, n_steps, n_dim])
    for i in range(tensor.shape[1]):
        mat = tensor[:,i,:]
        max_val = np.max(mat, 0) # shape: n_dim
        min_val = np.min(mat, 0)
        mat_norm = (mat - min_val) / (max_val - min_val + 1e-12)

        tensor_norm[:,i,:] = mat_norm

    # print norm_x
    return tensor_norm

def meanstd_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    #tensor_norm = np.ones([n_node, n_steps, n_dim])
    tensor_reshape = preprocessing.scale(np.reshape(tensor, [n_node, n_steps*n_dim]), axis=1)
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])
        
    # print norm_x
    return tensor_norm

# def get_Batch(data, label, batch_size, n_epochs):
#     #print(data.shape, label.shape)
#     input_queue = tf.train.slice_input_producer([data, label], num_epochs=n_epochs, shuffle=True, capacity=1000)
#     x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=1000, allow_smaller_final_batch=True)
#     return x_batch, y_batch


def Neightbor_aggre(node_attr, k): #node_attr: N_tr*50*42
    N, T, fea = np.shape(node_attr)
    for i in range(T):
        frame = node_attr[:,i,:]
        nbrs  = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(frame)
        _, indices = nbrs.kneighbors() #N_tr*k; '()' will not include the node itself as its neighbor
        frame_nbrs = frame[indices] #N_tr*k*42
        aggr_frame = np.mean(frame_nbrs, axis=1) #N_tr*42
        if i == 0:
            aggr = aggr_frame
        else:
            aggr = np.stack((aggr, aggr_frame), axis=1) 

    return aggr #N_tr*50*42

def generate_graph(fea_mat, n_nbor):
    # euclidean equals to pearson corrcoef after normalization
    graph = kneighbors_graph(fea_mat, n_nbor, mode='connectivity', metric='euclidean', include_self=False).toarray()

    return graph

def construct_adj(graph, n_sample):
    # graph: n_node*n_node
    # '1' indicates connection, '0' for no-connection
    # return adj: n_node*n_sample
    n_node = graph.shape[0]
    adj = (-1)*np.ones((n_node, n_sample))
    for i in range(n_node):
        idx_nbor = np.nonzero(graph[i])[0]
        if len(idx_nbor) == 0:
            continue
        if len(idx_nbor) > n_sample:
            idx_nbor = np.random.choice(idx_nbor, n_sample, replace=False)
        elif len(idx_nbor) < n_sample:
            idx_nbor = np.random.choice(idx_nbor, n_sample, replace=True)
        adj[i,:] = idx_nbor

    return adj

def sample_tf(inputs, n_layers, sample_sizes, adj_tensor, n_steps):
    # inputs: batch of nodes; adj_tensor: adj_mats for all time steps
    # return: 
    #   samples_tensor    : (n_steps)*inputs*(1+[0]+[1])
    #   support_sizes_list: a list of lists

    support_sizes_list = []
    for t in range(n_steps):
        samples = inputs
        input_temp = inputs
        support_sizes = []
        adj_mat = adj_tensor[t]

        # print(adj_mat)
        # print(inputs)

        support_size = 1

        samples_frame = torch.reshape(input_temp, [-1, 1])
        samples_frame = samples_frame.type(torch.int32)
        for i in range(n_layers):
            support_size *= sample_sizes[i]
            support_sizes.append(support_size)
            samples = samples.type(torch.int32)
            neighs = torch.select(adj_mat, samples)  # shape: samples*n_sample
            random.shuffle(torch.transpose(neighs).numpy())  # shuffle columns

            samples = torch.reshape(neighs, [-1])
            samples_frame = torch.concat([samples_frame, torch.reshape(neighs, [-1, support_sizes[i]])],
                                         1)  # shape: inputs*support_sizes

        if t == 0:
            samples_tensor = torch.unsqueeze(samples_frame,
                                             0)  # shape: 1*inputs*(1+support_sizes[0]+support_sizes[1])
        samples_tensor = torch.concat([samples_tensor, torch.unsqueeze(samples_frame, 0)],
                                      0)  # shape: (1+1)*inputs*(1+[0]+[1])

        support_sizes_list.append(support_sizes)

    return samples_tensor, support_sizes_list


def normalize(v):
    return F.normalize(v, dim=1, p=2, eps=1e-12)

def eval(y_true, y_pred):
    # both size: 6400*num_classes
    
    # ACC
    correct_pred = np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1))
    accuracy = np.mean(correct_pred.astype(float))

    # AUC
    auc_mac = roc_auc_score(y_true, y_pred, average='macro')
    auc_mic = roc_auc_score(y_true, y_pred, average='micro')
    
    # F1
    f1_mac = f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='macro')
    f1_mic = f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='micro')


    return accuracy, auc_mac, auc_mic, f1_mac, f1_mic

def generate_topo(Graphs, dim_topo, prob_c, step_K):
    # Graphs: n_time*n_node*n_node
    # dim_topo:
    # prob_c:

    n_time, n_node = Graphs.shape[0], Graphs.shape[1]
    P_T = np.zeros((n_node,n_time,n_node))

    for t in range(n_time):
        print('time: ', t)
        Gt = Graphs[t,:,:]
        Gt = Gt + np.eye(n_node)
        #Dt = np.diag(np.sum(Gt, axis=1))
        Dt_inv = np.diag(1/np.sum(Gt, axis=1))
        Gt_norm = np.dot(Dt_inv, Gt) # n_node*n_node

        Pt0 = np.eye(n_node)
        Pt_p = Pt0
        for i in range(step_K):
            Pt_c = prob_c * np.dot(Pt_p, Gt_norm) + (1-prob_c) * Pt0 # n_node*n_node
            P_T[:,t,:] += Pt_c

            Pt_p = Pt_c
    
    # # PCA
    # Topology = np.zeros((n_node,n_time,dim_topo))
    # for t in range(n_time):
    #     print(t)
    #     pca = PCA(n_components=dim_topo)
    #     Topology[:,t,:] = pca.fit_transform(P_T[:,t,:])

    '''
    # LDA
    Topology = np.zeros((n_node,n_time,dim_topo))
    for t in range(n_time):
        lda = LatentDirichletAllocation(n_components=dim_topo)
        Topology[:,t,:] = lda.fit_transform(P_T[:,t,:])
    '''

    return P_T #(n_node, n_time, dim_topo)

def reduce_dim(topo_orgi, dim_topo):
    n_node, n_time, n_node = np.shape(topo_orgi)
    Topology = np.zeros((n_node, n_time, dim_topo))
    
    # PCA
    for t in range(n_time):
        print(t)
        pca = PCA(n_components=dim_topo)
        Topology[:,t,:] = pca.fit_transform(topo_orgi[:,t,:])

    '''
    # LDA
    for t in range(n_time):
        lda = LatentDirichletAllocation(n_components=dim_topo)
        Topology[:,t,:] = lda.fit_transform(P_T[:,t,:])
    '''

    return Topology #(n_node, n_time, dim_topo)

def sample_gumbel(shape):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, dtype=torch.float64)
        return -torch.log(torch.log(U + eps) + eps)


def gumbel_softmax_sample(adj, shape, logits, temperature, istrain=True):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    r = sample_gumbel(logits.values.shape)
    r = r.type(torch.float64)
    values = lambda: torch.log(logits.values.type(torch.float64)) + r if istrain else lambda: torch.log(logits.values.type(torch.float64))
    # torch.where(istrain, lambda: math.log(logits.values.type(torch.float64)) + r,
    #              lambda: math.log(logits.values.type(torch.float64)))
    values /= temperature
    print(values, type(values))
    y = torch.sparse_coo_tensor(adj.coalesce().indices, values, shape)
    return torch.sparse.softmax(y)


if __name__ == '__main__':
    filedpath = '/Users/norahc/Downloads/Ada-Code-Data-ICDM2019/Data/brain/Brain_5000nodes.npz'
    file      = np.load(filedpath)

    Features  = file['attmats'] #(n_node, n_time, att_dim)
    print(Features.shape)
    f = rearrange(Features, 'b h w -> b (h w)')
    Labels    = file['labels']  #(n_node, num_classes)
    Graphs    = file['adjs']    #(n_time, n_node, n_node)

    Features = meanstd_normalization_tensor(Features)

    #knn = 5
    dim_redu  = 5000 # should be set to the same size as 'att_dim'
    n_hidden_units = 10
    n_hop = 2
    n_sample = 4
    sample_sizes = [n_sample, n_sample]
    n_nbor = 40

    n_node, n_steps, n_dim = np.shape(Features)
    num_classes = 10
    tmp_dim = num_classes

    batch_size = 3000
    # learning rate
    lr1 = 0.001
    lr2 = 0.0001
    lr3 = 0.00025
    training_iters = 500 # number of epochs
    num_stacked_layers = 1
    display_step = 1
    in_keep_prob  = 1 #0.5
    out_keep_prob = 1
    lambda_l2_reg = 5e-5

    M  = n_hidden_units
    # temporal attention
    r  = 1
    L  = 40
    L_tp = M
    # spatial attention
    L10 = n_dim
    L11 = 40
    # dims of hops
    dim_hop0 = L11
    dim_hop1 = L11*2

    # denoise
    eps = 1e-8
    k = 5

    # topology
    prob_c = 0.98
    step_K = 5
    dim_topo = n_dim # dim_topo will not work
    Topology = generate_topo(Graphs, dim_topo, prob_c, step_K) #(n_node, n_time, dim_topo)
    Topology = meanstd_normalization_tensor(Topology)

    # add self-loop
    for i in range(n_steps):
        Graphs[i,:,:] += np.eye(n_node, dtype=np.int32)
    model = Model(Features, Topology, n_hidden_units)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0.001, T_0=20)
    train_losses = []
    val_losses = []
    for epoch in range(200):
        Data_idx = np.arange(n_node)
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(Data_idx, Labels, test_size=0.1)  # N_tr, N_te
        X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size=0.1)  # N_tr, N_te

        train_dataset = MyDataset(X_train_idx, y_train)

        val_dataset = MyDataset(X_val_idx, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=1,
                                                   )

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=1,
                                                 )

        # construct adjacent matrix
        adj_tensor = np.ones((n_steps, n_node, n_sample), dtype=np.int32)

        for i in range(n_steps):
            adj = construct_adj(Graphs[i], 240)
            print('former adj', adj.shape)

            # denoise
            # adj = sparse.csr_matrix(adj)
            if not sparse.isspmatrix_coo(adj):
                adj = sparse.coo_matrix(adj)
            adj = adj.astype(np.float64)
            print(type(adj))
            indices = np.vstack(
                (adj.row, adj.col))
            shape = adj.shape
                # adj = (indices, adj.data, adj.shape)
            print('indices: ', indices, type(indices), indices.shape)
            print('data: ', adj.data, type(adj.data), len(adj.data))
            print('shape: ', adj.shape, type(adj.shape))
            adj = torch.sparse_coo_tensor(torch.tensor(indices, dtype=torch.int64), adj.data, adj.shape).type(torch.float32)
            print(f.shape)
            indice1 = torch.tensor(indices[1, :], dtype=torch.int64)
            print('i1: ', indice1.shape)
            indice0 = torch.tensor(indices[0, :], dtype=torch.int64)
            print('i2: ', indice0.shape)
            f1 = torch.gather(torch.Tensor(f), 0, indice1)
            f2 = torch.gather(torch.Tensor(f), 0, indice0)
            auv = torch.unsqueeze(adj.data, -1)
            auv = auv.type(torch.float64)
            temp = torch.concat([f1, f2, auv], -1)
            print('temp shape: ', temp.shape)
            in_features = temp.shape[-1]
            mlp3 = MLP_3(in_features)
            temp = mlp3(temp)
            z = torch.reshape(temp, [-1])
            z_matrix = torch.sparse_coo_tensor(indices, z, shape)
            pi = torch.sparse.softmax(z_matrix)

            y = gumbel_softmax_sample(adj, shape, pi, 1, True)
            y_dense = y.to_dense()
            print("dense shape:", y_dense.shape)

            top_k_v, top_k_i = torch.topk(y_dense, k)
            kth = torch.min(top_k_v, -1) + eps  # N,

            kth = torch.unsqueeze(kth, -1)
            kth = torch.tile(kth, [1, n_sample])  # N,N

            print("y_dense: ", y_dense.shape)
            print("kth: ", kth.shape)
            mask2 = torch.ge(y_dense, kth)
            mask2 = mask2.type(torch.float64)
            dense_support = mask2
            dense_support = torch.multiply(y_dense, mask2)
            self_edge = torch.eye(shape[0], n_sample, dtype=torch.float64)
            dense_support = dense_support + self_edge

            rowsum = torch.sum(dense_support, -1) + 1e-6  # to avoid NaN

            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])  # D^-0.5
            d_mat_inv_sqer = torch.diag(d_inv_sqrt)
            ad = torch.matmul(d_mat_inv_sqer, dense_support)
            print(d_mat_inv_sqer.shape, ad.shape)
            dadt = torch.matmul(d_mat_inv_sqer, ad)

            array = np.array(dadt)

            adj_tensor[i, :, :] = array


        adj_tensor = torch.tensor(adj_tensor, dtype=torch.int32)
        train_loader = tqdm(train_loader, file=sys.stdout)

        # training
        for step, data in enumerate(train_loader):
            inputs, label = data
            samples_idx, support_sizes = sample_tf(inputs, n_hop, sample_sizes, adj_tensor,
                                                   n_steps)  # samples_idx: (n_steps)*inputs*(1+[0]+[1])

            # 6400*num_classes, 6400*T*M, 6400*r*T, 6400*r*M, T*6400*[0]*(1+[1]/[0])

            model.add_parameter(samples_idx, support_sizes)
            logits_batch, h_list_batch, Alpha, embedding_att, Beta, Gamma, h_list_batch_topo = model(inputs)
            # logits_batch, h_list_batch, Alpha, embedding_att, Beta, Gamma, h_list_batch_topo = mi_gru(Features, Topology,
            #                                                                                           inputs,
            #                                                                                           samples_idx,
            #                                                                                           support_sizes,
            #                                                                                           n_hidden_units,
            #                                                                                           num_stacked_layers,
            #                                                                                           tmp_dim)
            prediction = F.softmax(logits_batch)  # softmax row by row, 6400*num_classes
            # L2 regularization for weights and biases
            # lambda_l2_reg = 5e-5
            reg_loss = 0


            output1 = torch.mean(torch.norm(torch.tensor(pg)))

            lambda_reg_att = 0e-1
            reg_att_temp = torch.matmul(Alpha, torch.permute(Alpha, (0, 2, 1)))  # 6400*r*r
            I_mat = torch.Tensor(np.eye(r), dtype=torch.float64)
            reg_att = torch.mean(torch.norm(reg_att_temp - I_mat))

            loss_p = F.cross_entropy(logits_batch, label)

            loss_op = torch.mean(loss_p)

            train_loss = loss_op + lambda_l2_reg * reg_loss + lambda_reg_att * reg_att

            train_losses.append(train_loss)

            train_loss.backward()

            scheduler.step()

        # evaluation
        model.eval()
        for step, data in enumerate(val_loader):
            inputs, label = data
            logits_batch, h_list_batch, Alpha, embedding_att, Beta, Gamma, h_list_batch_topo = model(inputs)
            # logits_batch, h_list_batch, Alpha, embedding_att, Beta, Gamma, h_list_batch_topo = mi_gru(Features, Topology,
            #                                                                                           inputs,
            #                                                                                           samples_idx,
            #                                                                                           support_sizes,
            #                                                                                           n_hidden_units,
            #                                                                                           num_stacked_layers,
            #                                                                                           tmp_dim)
            prediction = F.softmax(logits_batch)  # softmax row by row, 6400*num_classes
            # L2 regularization for weights and biases
            # lambda_l2_reg = 5e-5
            reg_loss = 0

            output1 = torch.mean(torch.norm(torch.tensor(pg)))

            lambda_reg_att = 0e-1
            reg_att_temp = torch.matmul(Alpha, torch.permute(Alpha, (0, 2, 1)))  # 6400*r*r
            I_mat = torch.Tensor(np.eye(r), dtype=torch.float64)
            reg_att = torch.mean(torch.norm(reg_att_temp - I_mat))

            loss_p = F.cross_entropy(logits_batch, label)

            loss_op = torch.mean(loss_p)

            val_loss = loss_op + lambda_l2_reg * reg_loss + lambda_reg_att * reg_att

            val_losses.append(val_loss)
