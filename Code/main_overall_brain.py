# encoding: utf-8
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] = "0,1"
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import LocallyLinearEmbedding
import random
from torch.autograd import Variable


class GraphDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data, label


def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)


def merge_list(list1, list2):
    return list1 + list2


def temporal_attention(h_list_tensor):
    # h_list_tensor: 6400*T*2M
    # V: L*2M
    # W: r*L
    # M: dh
    # r: da


    # A = torch.transpose(softmax(torch.matmul(W, torch.multiply(tanh(torch.matmul(V, torch.transpose(h_list_tensor))),
    #                                                            tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    transfer = torch.einsum('ij,jkl->ikl', V, torch.permute(h_list_tensor, (2, 1, 0)))  # FC layer
    temp = tanh(transfer)  # L*T*6400, tanh
    A = softmax(torch.einsum('ij,jkl->ikl', W, temp), dim=1)  # r*T*6400; FC layer and softmax

    return torch.permute(A, (2, 0, 1)) #6400*r*T


# Both are Attribute-Topology Attention
def topo_att_indi_attention(h_1, h_2):
    # h_1, h_2: 6400*T*M
    # V_tp: L_tp*M
    # w_tp: 1*L_tp

    T = h_1.size()[1].value

    for i in range(T):
        h_comb = torch.concat([torch.unsqueeze(h_1[:, i, :], 1), torch.unsqueeze(h_2[:, i, :], 1)], 1)  # 6400*2*M

        temp = tanh(torch.einsum('ij,jkl->ikl', V_tp, torch.permute(h_comb, (2, 1, 0))))  # L_tp*2*6400
        if i == 0:
            A = softmax(torch.einsum('ij,jkl->ikl', w_tp, temp), dim=1)  # 1*2*6400
        else:
            A = torch.concat([A, softmax(torch.einsum('ij,jkl->ikl', w_tp, temp), dim=1)], 0)  # T*2*6400

    return torch.permute(A, (2, 1, 0))  # 6400*2*T


def topo_att_all_attention(h_1, h_2):
    # h_1, h_2: 6400*T*M
    # V_tp: L_tp*TM
    # w_tp: 1*L_tp

    T = h_1.size()[1].value
    M = h_1.size()[2].value

    h1_tmp, h2_tmp = torch.reshape(h_1, [-1, T*M]), torch.reshape(h_2, [-1, T*M])  # 6400*TM
    h_comb = torch.concat([torch.unsqueeze(h1_tmp, 1), torch.unsqueeze(h2_tmp, 1)], 1)  # 6400*2*TM
    temp = tanh(torch.einsum('ij,jkl->ikl', V_tp, torch.permute(h_comb, (2, 1, 0))))  # L_tp*2*6400
    A = softmax(torch.einsum('ij,jkl->ikl', w_tp, temp), dim=1)  # 1*2*6400

    return torch.permute(A, (2, 1, 0))  # 6400*2*1


def normalize(v):
    return F.normalize(v, dim=1, eps=1e-12)
    # return tf.nn.l2_normalize(v, axis=1, epsilon=1e-12)



def mi_gru(Fea_batch, Topo_batch, idx_batch, samples_idx_batch, support_sizes_batch, n_hidden_units, num_stacked_layers, weights, biases, t_dim):
    # weights['out']: (r*2M)*n_classes
    # biases['out']:  n_classes

    h_list_tensor, Beta = GRU_Neighbor(Fea_batch, idx_batch, samples_idx_batch, support_sizes_batch, n_hidden_units, num_stacked_layers) #6400*T*M; T*6400*[0]*(1+[1]/[0])
    h_list_tensor_topo = GRU_Topology(Topo_batch, idx_batch, n_hidden_units) #6400*T*M

    Gamma = topo_att_all_attention(h_list_tensor, h_list_tensor_topo) #6400*2*1

    h_1_tmp = torch.mul(torch.permute(torch.unsqueeze(Gamma[:, 0, :], 1), (0, 2, 1)), h_list_tensor)  # 6400*1*1 * 6400*T*M = 6400*T*M
    h_2_tmp = torch.mul(torch.permute(torch.unsqueeze(Gamma[:, 1, :], 1), (0, 2, 1)), h_list_tensor_topo)  # 6400*1*1 * 6400*T*M = 6400*T*M

    h_att_topo = torch.concat([h_1_tmp, h_2_tmp], 2)  # 6400*T*2M

    print(h_att_topo)

    # apply temporal attention
    Alpha = temporal_attention(h_att_topo) #6400*r*T
    print(Alpha)

    temp  = torch.matmul(Alpha, h_att_topo) #6400*r*2M
    temp1 = torch.reshape(temp, [-1, temp.size()[1].value*temp.size()[2].value.type(torch.int32)])  # (6400)*(r*2M)

    return torch.matmul(temp1, weights['out']) + biases['out'], h_list_tensor, Alpha, temp, Beta, Gamma, h_list_tensor_topo #6400*num_classes


def GRU_Topology(Topo, idx, n_h_units):
	# Topo: 10000*50*dim_topo
	# idx: 6400

    ## GRU_Topo
    # Wz_tp   : 20*(20+42);   bz_tp : 20
    # Wr1_tp  : 20*(20+42);   br1_tp: 20
    # Wh_1_tp : 20*20;
    # Wh_2_tp : 20*42;
    # Wh   : [Wh_1,Wh_2];  bh_tp : 20

    T     = Topo.size()[1].value # T = 50
    n_dim = Topo.size()[2].value # n_dim = 42
    idx   = idx.type(torch.int32)
    X     = torch.gather(Topo, idx) # shape: 6400*50*42

    for i in range(T):
        #print("index i in LSTM:", i)
        xt_temp = X[:, i, :] # 6400*42
        xt      = torch.transpose(xt_temp) # 42*6400

        if i == 0:
            #print(Wh_2_tp)
            #print(xt)
            ht_temp  = torch.matmul(Wh_2_tp, xt) # (20*42)*(42*6400)=20*6400
            ht_tilde = torch.transpose(tanh(torch.transpose(ht_temp) + bh_tp)) # 6400*20 + 20 -> 20*6400
            ht       = ht_tilde

            hp       = ht #20*6400
            h_list_tensor = torch.unsqueeze(hp, 1)  # 20*1*6400

        else:
            Wh    = torch.concat([Wh_1_tp,Wh_2_tp], 1)   # [20*20,20*42] -> 20*62

            Wzr   = torch.concat([Wz_tp, Wr1_tp], 0)  # (20+20)*62
            bzr   = torch.concat([bz_tp, br1_tp], 0)  # (20+20)

            zr_temp = torch.matmul(Wzr, torch.concat([hp, xt], 0)) # (40*62)*((20+42)*6400)=40*6400
            zr      = torch.transpose(sigmoid(torch.transpose(zr_temp) + bzr)) # (20+20)*6400: zt, rt1

            rt1      = zr[n_h_units:n_h_units*2, :] # 20*6400
            ht_temp  = torch.matmul(Wh, torch.concat([torch.mul(rt1, hp), xt], 0)) # 20*6400
            ht_tilde = torch.transpose(tanh(torch.transpose(ht_temp) + bh_tp)) # 6400*20 + 20 -> 20*6400

            zt       = zr[0:n_h_units,:] # 20*6400
            zt_neg   = 1.0 - zt
            ht       = torch.mul(zt_neg, hp) + torch.mul(zt, ht_tilde)

            hp       = ht # 20*6400
            h_list_tensor = torch.concat([h_list_tensor, torch.unsqueeze(hp, 1)], 1)  # 20*T*6400

    h_list_tensor = torch.permute(h_list_tensor, (2, 1, 0)) # 6400*T*20 (M=20)

    return h_list_tensor


def GRU_Neighbor(Fea, idx, saps_idx, sup_sizes, n_h_units, num_stacked_layers):  #X_Nebr, n_h_units, num_stacked_layers
    # Fea: 10000*50*42, features
	# idx: 6400
	# saps_idx: 50*6400*(1+[0]+[1])
	# sup_sizes: [[[0],[1]],...,[[0],[1]]]

	# X_Nebr: 6400*50*(42+42) ('6400' is dynamic)
	# n_dim = 42
	# n_h_units = 168

    ## GRU
    # Wz   : 20*(20+42*2);   bz : 20
    # Wr1  : 20*(20+42*2);   br1: 20
    # Wr2_1: 42*20;
    # Wr2_2: 42*(42*2);
    # Wr2  : [Wr2_1,Wr2_2];  br2: 42
    # Wh_1 : 20*20;
    # Wh_2 : 20*(42*2);
    # Wh   : [Wh_1,Wh_2];    bh : 20

    #d = tf.cast(n_hidden_units/n_dim, tf.int32) # d = 4
    T     = Fea.size()[1].value  # T = 50
    n_dim = Fea.size()[2].value  # n_dim = 42
    idx = idx.type(torch.int32)
    X     = torch.gather(Fea, idx)  # shape: 6400*50*42
    #XN    = X_Nebr[:,:,n_dim:n_dim*2] # 6400*50*42

    #print(len(sup_sizes))

    for i in range(T):
        #print("index i in LSTM:", i)
        xt_temp  = X[:, i, :]  # 6400*42
        xt       = torch.permute(xt_temp)  # 42*6400
        
        # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
        xnt, Beta_step = aggregate_att_mean(Fea[:, i, :], saps_idx[i, :, :], sup_sizes[i], idx) # shape: 42*6400; 6400*[0]*(1+[1]/[0])

        if i == 0:
            rt2_temp = torch.matmul(Wr2_2, torch.concat([xt, xnt], 0)) # (42*84)*((42+42)*6400)=42*6400
            rt2      = torch.transpose(sigmoid(torch.transpose(rt2_temp) + br2))  # 6400*42 + 42 -> 42*6400

            ht_temp  = torch.matmul(Wh_2, torch.concat([xt, torch.mul(rt2, xnt)], 0))  # (20*84)*(84*6400)=20*6400
            ht_tilde = torch.transpose(tanh(torch.transpose(ht_temp) + bh)) # 6400*20 + 20 -> 20*6400
            ht       = ht_tilde

            hp       = ht #20*6400
            h_list_tensor = torch.unsqueeze(hp, 1) # 20*1*6400

            Beta = torch.unsqueeze(Beta_step, 0) # 1*6400*[0]*(1+[1]/[0])

        else:
            Wr2   = torch.concat([Wr2_1, Wr2_2], 1) # [42*20,42*84] -> 42*104
            Wh    = torch.concat([Wh_1, Wh_2], 1)   # [20*20,20*84] -> 20*104
            Wzr   = torch.concat([Wz, Wr1, Wr2], 0)  # (20+20+42)*104
            bzr   = torch.concat([bz, br1, br2], 0)  # (20+20+42)

            zr_temp = torch.matmul(Wzr, torch.concat([hp, xt, xnt], 0)) # (82*104)*((20+42+42)*6400)=82*6400
            zr      = torch.transpose(sigmoid(torch.transpose(zr_temp) + bzr)) # (20+20+42)*6400: zt, rt1, rt2

            rt1      = zr[n_h_units:n_h_units*2, :] # 20*6400
            rt2      = zr[n_h_units*2:n_h_units*2+n_dim, :] # 42*6400
            ht_temp  = torch.matmul(Wh, torch.concat([torch.mul(rt1, hp), xt, torch.mul(rt2, xnt)], 0)) # 20*6400
            ht_tilde = torch.transpose(tanh(torch.transpose(ht_temp) + bh)) # 6400*20 + 20 -> 20*6400

            zt       = zr[0:n_h_units,:] # 20*6400
            zt_neg   = 1.0 - zt
            ht       = torch.mul(zt_neg, hp) + torch.mul(zt, ht_tilde)

            hp       = ht # 20*6400
            h_list_tensor = torch.concat([h_list_tensor, torch.unsqueeze(hp, 1)], 1) # 20*T*6400

            Beta = torch.concat([Beta, torch.unsqueeze(Beta_step, 0)], 0)  # (T)*6400*[0]*(1+[1]/[0])

    h_list_tensor = torch.permute(h_list_tensor, (2, 1, 0)) # 6400*T*20 (M=20)

    return h_list_tensor, Beta


def aggregate_mean(fea_mat,  samples,  sup_sizes, targets):
    # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
	# return xnt: 42*6400

    sup_sizes = [1] + sup_sizes # [1, [0], [1]]
    # for i in range(len(sup_sizes)-1,0,-1): # '-1 0 -1'
    i = 2
    dim_temp = np.int32(sup_sizes[i]/sup_sizes[i-1])
    samples_temp = torch.reshape(samples[:, -sup_sizes[i]:], [-1, sup_sizes[i-1], dim_temp])  # shape: 6400*[0]*([1]/[0])
    fea_mat_samples_temp = torch.index_select(fea_mat, 0, samples_temp)  # shape: 6400*[0]*([1]/[0])*42
    fea_hop_mean = torch.mean(fea_mat_samples_temp, 2)  # shape: 6400*[0]*42

    # fea_mat_samples_temp_1 = tf.sigmoid(tf.matmul(weights_hops[str(i-1)], tf.transpose(fea_hop_mean, perm=[2, 1, 0])))  # shape: (42,42)*(42*[0]*6400)=42*[0]*6400
    fea_mat_samples_temp_1 = sigmoid(torch.einsum('ij,jkl->ikl', weights_hops[str(i-1)], torch.permute(fea_hop_mean, (2, 1, 0)))) # shape: (42,42)*(42*[0]*6400)=42*[0]*6400
    fea_hop_mean_1 = torch.mean(torch.permute(fea_mat_samples_temp_1, (2, 1, 0)), 1)  # shape: 6400*42
    res = sigmoid(torch.matmul(weights_hops[str(i-2)], torch.permute(fea_hop_mean_1, (1, 0))))  # shape: (42*42)*(42*6400)=42*6400

    return res


def spatial_attention_1(f_tar_samp):
    # f_tar_samp: 6400*[0]*([1]/[0])*84; 6400*1*[0]*84
    # V1_h1: L11*n_dim
    # w1_h1: (2*L11)
    temp_sp1 = torch.einsum('ij,jklm->iklm', V1_h1, torch.permute(f_tar_samp[:, :, :, 0:n_dim], (3, 2, 1, 0)))  # L11*([1]/[0])*[0]*6400
    temp_sp2 = torch.einsum('ij,jklm->iklm', V1_h1, torch.permute(f_tar_samp[:, :, :, n_dim:2*n_dim], (3, 2, 1, 0)))  # L11*([1]/[0])*[0]*6400
    B = softmax(LReLU(torch.einsum('j,jklm->klm', w1_h1, torch.concat([temp_sp1, temp_sp2], 0))), dim=0)  #([1]/[0])*[0]*6400

    return torch.transpose(B, (2, 1, 0))  # 6400*[0]*([1]/[0]); 6400*1*[0]


def spatial_attention_0(f_tar_samp):
    # f_tar_samp: (2*dim_hop0)*[0]*6400
    # V1_h0: L10*L11
    # w1_h0: (2*L10)
    # dim_hop0 = L11
    
    temp_sp1 = torch.einsum('ij,jkl->ikl', V1_h0, f_tar_samp[0:dim_hop0, :, :]) # L10*[0]*6400
    temp_sp2 = torch.einsum('ij,jkl->ikl', V1_h0, f_tar_samp[dim_hop0:2*dim_hop0, :, :]) # L10*[0]*6400
    B = softmax(LReLU(torch.einsum('j,jkl->kl', w1_h0, torch.concat([temp_sp1, temp_sp2], 0))), dim=0)  # [0]*6400

    return torch.permute(B, (1, 0))  # 6400*[0]

def aggregate_att_mean(fea_mat,  samples, sup_sizes, targets):
    # shapes: 10000*42, 6400*(1+[0]+[1]), [[0],[1]], 6400
    # return xnt: 42*6400

    sup_sizes = [1] + sup_sizes  # [1, [0], [1]]

    ## k=1: update node representations of Hop-0
    dim_temp = np.int32(sup_sizes[2]/sup_sizes[1])
    # gather neighbors at hop-1
    samples_temp = torch.reshape(samples[:, -sup_sizes[2]:], [-1, sup_sizes[1], dim_temp]) # shape: 6400*[0]*([1]/[0])
    fea_mat_samples_temp = torch.index_select(fea_mat, samples_temp) # shape: 6400*[0]*([1]/[0])*42

    # gather target nodes for hop-1 and replicate attributes
    samples_temp_t = samples[:, -(sup_sizes[1]+sup_sizes[2]):-sup_sizes[2]] # shape: 6400*[0]
    target_temp = torch.unsqueeze(samples_temp_t, 2) # 6400*[0]*1
    target_temp = torch.tile(target_temp, [1, 1, dim_temp]) # 6400*[0]*([1]/[0])
    fea_mat_samples_temp_t = torch.index_select(fea_mat, target_temp)  # shape: 6400*[0]*([1]/[0])*42

    # learn attention for hop-1
    fea_tar_samp = torch.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 3)  # shape: 6400*[0]*([1]/[0])*84
    Beta_hop1 = spatial_attention_1(fea_tar_samp)  # shape: 6400*[0]*([1]/[0])

    # calculate neighbor representation
    #   V1_h1: L11*n_dim
    #   fea_mat_samples_temp: 6400*[0]*([1]/[0])*42
    temp_aam_0 = torch.einsum('ij,jklm->iklm', V1_h1, torch.permute(fea_mat_samples_temp, (3, 2, 1, 0))) # shape: L11*([1]/[0])*[0]*6400
    temp_aam_1 = torch.matmul(torch.unsqueeze(Beta_hop1, 2), torch.permute(temp_aam_0, (3, 2, 1, 0))) # shape: 6400*[0]*1*L11 ???
    fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=2) # shape: 6400*[0]*L11

    # concatenate and transform
    fea_mat_samples_temp_t_emb = torch.einsum('ij,jkl->ikl', V1_h1, torch.permute(fea_mat_samples_temp_t[:,:,0,:], (2, 1, 0)))  # L11*[0]*6400
    con_hop1 = torch.concat([fea_mat_samples_temp_t_emb, torch.permute(fea_hop_mean_att, (2, 1, 0))], 0)  # shape:(L11*2)*[0]*6400, dim_hop1=L11*2
    hop1 = sigmoid(torch.einsum('ij,jkl->ikl', weights_hops['1'], con_hop1))  # shape: (dim_hop0,dim_hop1)*((L11*2),[0]*6400)=dim_hop0*[0]*6400

    ## k=1: update node representations of nodes in batch
    dim_temp = np.int32(sup_sizes[1])
    # gather neighbors at hop-0
    samples_temp = samples[:, -(sup_sizes[1]+sup_sizes[2]):-sup_sizes[2]]  # shape: 6400*[0]
    fea_mat_samples_temp = torch.index_select(fea_mat, samples_temp)  # shape: 6400*[0]*42
    # gather target nodes for hop-0 and replicate attributes
    samples_temp_t = samples[:, 0]  # shape: 6400*1
    target_temp = torch.tile(torch.unsqueeze(samples_temp_t, 1), [1, dim_temp])  # 6400*[0]
    fea_mat_samples_temp_t = torch.index_select(fea_mat, target_temp)  # shape: 6400*[0]*42
    # learn attention for hop-0
    fea_tar_samp = torch.concat([fea_mat_samples_temp_t, fea_mat_samples_temp], 2) # shape: 6400*[0]*84
    Beta_hop0 = spatial_attention_1(torch.unsqueeze(fea_tar_samp, 1))  # shape: 6400*1*[0]
    # calculate neighbor representation
    #   V1_h1: L11*n_dim
    #   fea_mat_samples_temp: 6400*[0]*42
    temp_aam_0 = torch.einsum('ij,jkl->ikl', V1_h1, torch.permute(fea_mat_samples_temp, (2, 1, 0)))  # shape: L11*[0]*6400
    temp_aam_1 = torch.matmul(Beta_hop0, torch.permute(temp_aam_0, (2, 1, 0)))  # shape: 6400*1*L11 ???
    fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=1)  # shape: 6400*L110
    
    # concatenate and transform
    fea_mat_samples_temp_t_emb = torch.einsum('ij,jk->ik', V1_h1, torch.permute(fea_mat_samples_temp_t[:, 0, :], (1, 0)))  # L11*6400
    con_hop1 = torch.concat([fea_mat_samples_temp_t_emb, torch.permute(fea_hop_mean_att, (1, 0))], 0)  # shape: (L11*2)*6400, dim_hop1=L11*2
    hop0 = sigmoid(torch.einsum('ij,jk->ik', weights_hops['1'], con_hop1))  # shape: (dim_hop0,dim_hop1)*((L11*2)*6400)=dim_hop0*6400


    ## k=2: generate neighbor representation
    dim_temp = np.int32(sup_sizes[1])
    # gather target nodes for hop-0 and replicate attributes
    target_temp = torch.unsqueeze(hop0, 1) # dim_hop0*1*6400
    target_temp = torch.tile(target_temp, [1, dim_temp, 1]) # dim_hop0*[0]*6400
    # learn attention
    fea_tar_samp = torch.concat([target_temp, hop1], 0)  # shape: (2*dim_hop0)*[0]*6400
    Beta_hop = spatial_attention_0(fea_tar_samp)  # shape: 6400*[0]

    # calculate neighbor representation
    #   V1_h0: L10*dim_hop0
    #   hop1: dim_hop0*[0]*6400
    temp_aam_0 = torch.einsum('ij,jkl->ikl', V1_h0, hop1)  # shape: L10*[0]*6400
    temp_aam_1 = torch.matmul(torch.unsqueeze(Beta_hop, 1), torch.permute(temp_aam_0, (2, 1, 0)))  # shape: 6400*1*L10 ???
    fea_hop_mean_att = torch.squeeze(temp_aam_1, dim=1)  # shape: 6400*L10

    ## concatenate spatial attention
    # Beta_hop1: 6400*[0]*([1]/[0])
    # Beta_hop: 6400*[0]
    Beta_step = torch.concat([torch.unsqueeze(Beta_hop, 2), Beta_hop1], 2)  # 6400*[0]*(1+[1]/[0])

    return torch.transpose(fea_hop_mean_att), Beta_step  #L10*6400 = 42*6400; 6400*[0]*(1+[1]/[0])


def minmax_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    tensor_norm = np.ones([n_node, n_steps, n_dim])
    for i in range(tensor.shape[1]):
        mat = tensor[:, i, :]
        max_val = np.max(mat, 0) # shape: n_dim
        min_val = np.min(mat, 0)
        mat_norm = (mat - min_val) / (max_val - min_val + 1e-12)

        tensor_norm[:, i, :] = mat_norm

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
#     dataset = GraphDataset(data, label)
#     data_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

    return aggr  # N_tr*50*42

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
    # inputs: batch of nodes;
    # adj_tensor: adj_mats for all time steps
    # return: 
    #   samples_tensor    : (n_steps)*inputs*(1+[0]+[1])
    #   support_sizes_list: a list of lists

    support_sizes_list = []
    for t in range(n_steps):
        samples = inputs
        input_temp = inputs
        support_sizes = []
        adj_mat = adj_tensor[t]

        #print(adj_mat)
        #print(inputs)

        support_size = 1

        samples_frame = torch.reshape(input_temp, [-1, 1])
        samples_frame = samples_frame.type(torch.int32)
        for i in range(n_layers):
            support_size *= sample_sizes[i]
            support_sizes.append(support_size)
            samples = samples.type(torch.int32)
            neighs = torch.select(adj_mat, samples)  # shape: samples*n_sample
            neighs = torch.transpose(neighs)
            index = [i for i in len(neighs)]
            random.shuffle(index)
            neighs = neighs[index]  # shuffle columns

            samples = torch.reshape(neighs, [-1])
            samples_frame = torch.concat([samples_frame, torch.reshape(neighs, [-1, support_sizes[i]])], 1)  # shape: inputs*support_sizes
            
        if t == 0:
            samples_tensor = torch.unsqueeze(samples_frame, 0)  # shape: 1*inputs*(1+support_sizes[0]+support_sizes[1])
        samples_tensor = torch.concat([samples_tensor, torch.unsqueeze(samples_frame, 0)], 0)  # shape: (1+1)*inputs*(1+[0]+[1])

        support_sizes_list.append(support_sizes)

    return samples_tensor, support_sizes_list


def normalize(v):
    return torch.nn.functional.normalize(v, p=2, dim=1, eps=1e-12)


def truncated_normal_(tensor, mean=-0.1, std=0.1):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


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
        print(t)
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
        Topology[:, t, :] = pca.fit_transform(topo_orgi[:, t, :])

    '''
    # LDA
    for t in range(n_time):
        lda = LatentDirichletAllocation(n_components=dim_topo)
        Topology[:,t,:] = lda.fit_transform(P_T[:,t,:])
    '''

    return Topology #(n_node, n_time, dim_topo)


if __name__ == '__main__':
    tanh = torch.nn.Tanh()
    softmax = torch.nn.Softmax()
    sigmoid = torch.nn.Sigmoid()
    LReLU = torch.nn.LeakyReLU()
    cross_entropy = torch.nn.CrossEntropyLoss()
    filedpath = '/.../Data'
    filename  = '/brain/Brain_5000nodes.npz'
    file      = np.load(filedpath+filename)

    Features  = file['attmats'] #(n_node, n_time, att_dim)
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
    lambda_l2_reg = 5e-5 # regularization term

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

    # extract topology
    prob_c = 0.98
    step_K = 5
    dim_topo = n_dim # dim_topo will not work
    Topology = generate_topo(Graphs, dim_topo, prob_c, step_K) #(n_node, n_time, n_node)
    Topology = meanstd_normalization_tensor(Topology)

    # add self-loop
    for i in range(n_steps):
        Graphs[i,:,:] += np.eye(n_node, dtype=np.int32)

    Data_idx = np.arange(n_node)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(Data_idx, Labels, test_size=0.1) #N_tr, N_te; test_size=0.1
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size=0.1) #N_tr, N_te; test_size=0.1

    # tf.reset_default_graph()
    weights_att = {'W': truncated_normal_(torch.randn([r, L], dtype=torch.float64)),
                    'V': truncated_normal_(torch.randn([L, 2*M], dtype=torch.float64)), \
                        'w1_h1': truncated_normal_(torch.randn([2*L11], dtype=torch.float64)), \
                        'V1_h1': truncated_normal_(torch.randn([L11, n_dim], dtype=torch.float64)), \
                        'w1_h0': truncated_normal_(torch.randn([2*L10], dtype=torch.float64)), \
                        'V1_h0': truncated_normal_(torch.randn([L10, L11], dtype=torch.float64)), \
                            'w_tp': truncated_normal_(torch.randn([1, L_tp], dtype=torch.float64)), \
                            'V_tp': truncated_normal_(torch.randn([L_tp,n_steps*M], dtype=torch.float64))}
    
    # h_1, h_2: 6400*T*M
    # V_tp: L_tp*M
    # w_tp: 1*L_tp

    W, V = weights_att['W'], weights_att['V']
    w1_h0, V1_h0 = weights_att['w1_h0'], weights_att['V1_h0']
    w1_h1, V1_h1 = weights_att['w1_h1'], weights_att['V1_h1']
    w_tp, V_tp = weights_att['w_tp'], weights_att['V_tp']

    ## GRU_Neighbor
    # Wz   : 20*(20+42*2);   bz : 20
    # Wr1  : 20*(20+42*2);   br1: 20
    # Wr2_1: 42*20;
    # Wr2_2: 42*(42*2);
    # Wr2  : [Wr2_1,Wr2_2];  br2: 42
    # Wh_1 : 20*20;
    # Wh_2 : 20*(42*2);
    # Wh   : [Wh_1,Wh_2];    bh : 20

    Wz    = truncated_normal_(torch.randn([M, M+n_dim*2], dtype=torch.float64)) # 20*(20+42*2)
    Wr1   = truncated_normal_(torch.randn([M, M+n_dim*2], dtype=torch.float64)) # 20*(20+42*2)
    Wr2_1 = truncated_normal_(torch.randn([n_dim, M], dtype=torch.float64)) # 42*20
    Wr2_2 = truncated_normal_(torch.randn([n_dim, n_dim*2], dtype=torch.float64)) # 42*(42*2)
    Wh_1  = truncated_normal_(torch.randn([M, M], dtype=torch.float64)) # 20*20
    Wh_2  = truncated_normal_(torch.randn([M, n_dim*2], dtype=torch.float64)) # 20*(42*2)

    bz  = torch.zeros([M], dtype=torch.float64) # 20
    br1 = torch.zeros([M], dtype=torch.float64) # 20
    br2 = torch.zeros([n_dim], dtype=torch.float64) # 42
    bh  = torch.zeros([M], dtype=torch.float64) # 20

    ## GRU_Topo
    # Wz_tp   : 20*(20+42);   bz_tp : 20
    # Wr1_tp  : 20*(20+42);   br1_tp: 20
    # Wh_1_tp : 20*20;
    # Wh_2_tp : 20*42;
    # Wh   : [Wh_1_tp,Wh_2_tp];  bh_tp : 20

    Wz_tp    = truncated_normal_(torch.randn([M, M+dim_redu], dtype=torch.float64), mean=0)  # 20*(20+42)
    Wr1_tp   = truncated_normal_(torch.randn([M, M+dim_redu], dtype=torch.float64, ), mean=0)  # 20*(20+42)
    Wh_1_tp  = truncated_normal_(torch.randn([M, M], dtype=torch.float64), mean=0) # 20*20
    Wh_2_tp  = truncated_normal_(torch.randn([M, dim_redu], dtype=torch.float64), mean=0) # 20*42

    bz_tp  = truncated_normal_(torch.randn([M], dtype=torch.float64), mean=0)  # 20
    br1_tp = truncated_normal_(torch.randn([M, M], dtype=torch.float64), mean=0)  # 20
    bh_tp  = truncated_normal_(torch.randn([M, M], dtype=torch.float64), mean=0)  # 20

    #initializer=tf.constant_initializer(0.01)
    #initializer=tf.random_normal_initializer()
    # truncated_normal_initializer()shape=
    weights = {'out': truncated_normal_(torch.randn([r*2*M, num_classes], dtype=torch.float64))}
    biases  = {'out': torch.zeros([num_classes], dtype=torch.float64)}

    weights_hops = {'0': truncated_normal_(torch.randn([n_dim, dim_hop0], dtype=torch.float64)),
                    '1': truncated_normal_(torch.randn([dim_hop0, dim_hop1], dtype=torch.float64))}

    # placeholder
    x_idx       = tf.placeholder(tf.int64, [None,])
    y           = tf.placeholder(tf.int64, [None, num_classes])
    Features_tf = tf.placeholder(tf.float64, [n_node, n_steps, n_dim])
    Topology_tf = tf.placeholder(tf.float64, [n_node, n_steps, dim_redu])

    # construct adjacent matrix
    adj_tensor = np.ones((n_steps, n_node, n_sample), dtype=np.int32)
    for i in range(n_steps):
        adj_tensor[i, :, :] = construct_adj(Graphs[i], n_sample)
    adj_tensor = torch.from_numpy(adj_tensor).type(torch.int32)

    samples_idx, support_sizes = sample_tf(x_idx, n_hop, sample_sizes, adj_tensor, n_steps) # samples_idx: (n_steps)*inputs*(1+[0]+[1])

    #6400*num_classes, 6400*T*M, 6400*r*T, 6400*r*M, T*6400*[0]*(1+[1]/[0])
    logits_batch, h_list_batch, Alpha, embedding_att, Beta, Gamma, h_list_batch_topo = mi_gru(Features_tf, Topology_tf, x_idx, samples_idx, support_sizes, n_hidden_units, num_stacked_layers, weights, biases, tmp_dim) 
    prediction = softmax(logits_batch) #softmax row by row, 6400*num_classes

    # L2 regularization for weights and biases
    #lambda_l2_reg = 5e-5
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        reg_loss += torch.mean(torch.nn.functional.normalize(tf_var, p=2))

    # regularization for different hops of attentions
    # Alpha: 6400*r*20
    lambda_reg_att = 0e-1
    reg_att_temp = torch.matmul(Alpha, torch.permute(Alpha, (0, 2, 1)))  # 6400*r*r
    I_mat = torch.from_numpy(np.eye(r)).type(torch.float64)
    reg_att = torch.mean((reg_att_temp - I_mat))

    # Define loss and optimization
    # AdamOptimizer, GradientDescentOptimizer, AdagradOptimizer
    loss_op  = torch.mean(cross_entropy(logits_batch, y))
    
    train_op1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss_op + lambda_l2_reg*reg_loss + lambda_reg_att*reg_att)
    train_op2 = tf.train.AdamOptimizer(learning_rate=lr2).minimize(loss_op + lambda_l2_reg*reg_loss + lambda_reg_att*reg_att)
    train_op3 = tf.train.AdamOptimizer(learning_rate=lr3).minimize(loss_op + lambda_l2_reg*reg_loss + lambda_reg_att*reg_att)

    '''
	# L2 regularization for weights and biases
	reg_loss = 0
	for tf_var in tf.trainable_variables():
	    if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
	        reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))
	cost = clloss + 2.5*reg_loss
	train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    '''

    saver = tf.train.Saver()

    x_batch_idx, y_batch = get_Batch(X_train_idx, y_train, batch_size, training_iters)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            ite_tag = 0
            train_loss = 0
            List_tr_B_loss = []

            List_te_acc    = []
            List_te_auc1   = []
            List_te_auc2   = []
            List_te_f11    = []
            List_te_f12    = []

            List_tr_all_gamma = []

            while not coord.should_stop():
                x_batch_idx_feed, y_batch_feed= sess.run([x_batch_idx, y_batch])

                # training
                if ite_tag//(len(X_train_idx)//batch_size + 1) < 200:
                    train_op1.run(feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed, Features_tf:Features, Topology_tf:Topology})
                if (ite_tag//(len(X_train_idx)//batch_size + 1) >= 200) & (ite_tag//(len(X_train_idx)//batch_size + 1) < 400):
                    train_op2.run(feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed, Features_tf:Features, Topology_tf:Topology})
                if ite_tag//(len(X_train_idx)//batch_size + 1) >= 400:
                    train_op3.run(feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed, Features_tf:Features, Topology_tf:Topology})

                fetch = {'loss_op':loss_op, 'x_idx': x_idx, 'y':y, 'prediction':prediction, 'logits_batch':logits_batch, 'h_list_batch':h_list_batch, 'reg_loss':reg_loss, 'reg_att':reg_att, 'Alpha':Alpha, 'Beta':Beta, 'samples_idx':samples_idx, 'Gamma':Gamma, 'h_list_batch_topo':h_list_batch_topo} # Alpha: 6400*r*T; Beta: T*6400*[0]*(1+[1]/[0])
                
                # training result
                Res = sess.run(fetch, feed_dict={x_idx:x_batch_idx_feed, y:y_batch_feed, Features_tf:Features, Topology_tf:Topology})
                tr_acc, tr_auc1, tr_auc2, tr_f11, tr_f12 = eval(Res['y'], Res['prediction'])

                # validation resilt
                Res_val = sess.run(fetch, feed_dict={x_idx:X_val_idx, y:y_val, Features_tf:Features, Topology_tf:Topology})
                val_acc, val_auc1, val_auc2, val_f11, val_f12 = eval(Res_val['y'], Res_val['prediction'])

                # test result
                Res_test1 = sess.run(fetch, feed_dict={x_idx:X_test_idx, y:y_test, Features_tf:Features, Topology_tf:Topology})
                te1_acc, te1_auc1, te1_auc2, te1_f11, te1_f12 = eval(Res_test1['y'], Res_test1['prediction'])

                # result of whole training set
                Res_tr_all = sess.run(fetch, feed_dict={x_idx:X_train_idx, y:y_train, Features_tf:Features, Topology_tf:Topology})

                #print(Res['x_idx'])
                #print(np.argmax(Res['y'], 1))
                #print(np.argmax(Res['prediction'], 1))
                #print(Res['h_list_batch'])                
                #print(Res['logits_batch'])
                #print(Res['y'])
                #print(Res['prediction'])

                if ite_tag % display_step == 0 or ite_tag == 0:
                    print("Epoch %d, Ite %d, tr_loss=%g, L2=%g, L2_att=%g, val_loss=%g" % (ite_tag//(len(X_train_idx)//batch_size + 1), ite_tag, Res['loss_op'], Res['reg_loss'], Res['reg_att'], Res_val['loss_op']))
                    print("Epoch %d, Ite %d, tr_acc=%g, tr_auc1=%g, tr_auc2=%g, tr_f11=%g, tr_f12=%g, val_acc=%g, val_auc1=%g, val_auc2=%g, val_f11=%g, val_f12=%g" % (ite_tag//(len(X_train_idx)//batch_size + 1), ite_tag, tr_acc, tr_auc1, tr_auc2, tr_f11, tr_f12, val_acc, val_auc1, val_auc2, val_f11, val_f12))
    
                    # print(Res_tr_all['Alpha'][0:1]) # 6400*r*T
                    print("---------Gamma---------")
                    print(Res_tr_all['Gamma'][0:3]) # 6400*2*1
                    # print("---------Attri---------")
                    # print(Res_tr_all['h_list_batch'][0:3,0,:]) # 6400*T*M
                    # print("---------Topo--------")
                    # print(Res_tr_all['h_list_batch_topo'][0:3,0,:]) # 6400*T*M

                ite_tag += 1

                List_tr_B_loss = np.append(List_tr_B_loss, Res['loss_op'])
                List_te_acc    = np.append(List_te_acc, te1_acc)
                List_te_auc1   = np.append(List_te_auc1, te1_auc1)
                List_te_auc2   = np.append(List_te_auc2, te1_auc2)
                List_te_f11    = np.append(List_te_f11, te1_f11)
                List_te_f12    = np.append(List_te_f12, te1_f12)

                List_tr_all_gamma = np.append(List_tr_all_gamma, Res_tr_all['Gamma']) # [6400*2*1, ...]

        except tf.errors.OutOfRangeError:
            print("---Train end---")
        finally:
            coord.request_stop()
            print('---Programm end---')
        coord.join(threads)

        # testing
        Res_test = sess.run(fetch, feed_dict={x_idx:X_test_idx, y:y_test, Features_tf:Features, Topology_tf:Topology})
        te_acc, te_auc1, te_auc2, te_f11, te_f12 = eval(Res_test['y'], Res_test['prediction'])
        
        print("test_acc=%g, test_auc1=%g, test_auc2=%g, test_f11=%g, test_f12=%g" % (te_acc, te_auc1, te_auc2, te_f11, te_f12))

        # Alpha: 6400*r*T; Beta: T*6400*[0]*(1+[1]/[0]); Gamma: 6400*2*T
        # print('Temporal attention (Node-0):', Res_test['Alpha'][0,:,:])
        # print('Spatial attention (Node-0 at Step-0):', Res_test['Beta'][0,0,0,:])
        # print('Test ids:', X_test_idx)
        # print(Res_test['Gamma'][0:10]) #6400*2*1
        # samples_idx: (n_timestep)*6400*(1+[0]+[1])

        np.savez("/.../res_overall_brain.npz", List_tr_B_loss=List_tr_B_loss, List_te_acc=List_te_acc, List_te_auc1=List_te_auc1, List_te_auc2=List_te_auc2, List_te_f11=List_te_f11, List_te_f12=List_te_f12, Alpha=Res_test['Alpha'], Beta=Res_test['Beta'], X_test_idx=X_test_idx, y_test=y_test, samples_idx=Res_test['samples_idx'], Gamma=Res_test['Gamma'], List_tr_all_gamma=List_tr_all_gamma, Alpha_trall=Res_tr_all['Alpha'], Beta_trall=Res_tr_all['Beta'], samples_idx_trall=Res_tr_all['samples_idx'], y_train=y_train, X_train_idx=X_train_idx)


