import copy
import inspect
import torch
import torch.nn as nn
import torch_geometric
import numpy as np

def make_undirected(edge_index):
    return torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)

class Augment(object):
    def __init__(self, graphs, weak_aug, strong_aug):
        super().__init__()
        self.graphs = graphs
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.weak_graphs = []
        self.strong_graphs = []
        self.weak_data_list = []
        self.strong_data_list = []

        for graph in self.graphs:
            weak_graph = self.weak_aug(graph)
            strong_graph = self.strong_aug(graph)
            weak_graph.num_nodes = weak_graph.x.size(0)
            strong_graph.num_nodes = strong_graph.x.size(0)
            self.weak_graphs.append(weak_graph)
            self.strong_graphs.append(strong_graph)
            if strong_graph.x.shape[0] != strong_graph.num_nodes:
                print(strong_graph.x.shape[0], strong_graph.num_nodes)


    def __call__(self, idx):
        self.weak_data_list = []
        self.strong_data_list = []

        for i in idx:
            self.weak_data_list.append(self.weak_graphs[i])
            self.strong_data_list.append(self.strong_graphs[i])

        result = (
            torch_geometric.data.Batch.from_data_list(self.weak_data_list),
            torch_geometric.data.Batch.from_data_list(self.strong_data_list)
        )
        
        self.weak_data_list = []
        self.strong_data_list = []
        
        return result

def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)
    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
        data.num_nodes = data.x.size(0)
    except:
        data = data
    return data


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)
    data.num_nodes = data.x.size(0)
    return data

def subgraph(data, aug_ratio):
    if(aug_ratio == 0):
        return data
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = max(int(node_num * aug_ratio), 2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[list(range(node_num)), list(range(node_num))] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    data.edge_index = edge_index
    # data.num_nodes = data.x.size(0)
    # print(data.num_nodes)
    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)
    data.num_nodes = data.x.size(0)
    return data


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, data):
        return data
    
    def acr(self):
        return "Id"

class NodeDrop(nn.Module):
    def __init__(self, aug_ratio):
        super(NodeDrop, self).__init__()
        self.aug_ratio = aug_ratio

    def forward(self, data):
        return drop_nodes(data, self.aug_ratio)
    
    def acr(self):
        return "ND"


class Subgraph(nn.Module):
    def __init__(self, aug_ratio):
        super(Subgraph, self).__init__()
        self.aug_ratio = aug_ratio

    def forward(self, data):
        return subgraph(data, self.aug_ratio)
    
    def acr(self):
        return "Sg"
    
class EdgePert(nn.Module):
    def __init__(self, aug_ratio):
        super(EdgePert, self).__init__()
        self.aug_ratio = aug_ratio

    def forward(self, data):
        return permute_edges(data, self.aug_ratio)
    
    def acr(self):
        return "EP"


class AttrMask(nn.Module):
    def __init__(self, aug_ratio):
        super(AttrMask, self).__init__()
        self.aug_ratio = aug_ratio

    def forward(self, data):
        return mask_nodes(data, self.aug_ratio)
    
    def acr(self):
        return "AM"

class RandAugment(nn.Module):
    def __init__(self, num, ratio):
        super(RandAugment, self).__init__()
        self.num = num
        self.aug_ratio = ratio

        self.augmentations = {
            'drop_nodes': drop_nodes,
            'subgraph': subgraph,
            'permute_edges': permute_edges,
            'mask_nodes': mask_nodes
        }

        self.num_to_augmentations = {
            20: ['drop_nodes', 'subgraph'],
            21: ['drop_nodes', 'permute_edges'],
            22: ['drop_nodes', 'mask_nodes'],
            23: ['subgraph', 'permute_edges'],
            24: ['subgraph', 'mask_nodes'],
            25: ['permute_edges', 'mask_nodes'],
            30: ['drop_nodes', 'subgraph', 'permute_edges'],
            31: ['drop_nodes', 'subgraph', 'mask_nodes'],
            32: ['drop_nodes', 'permute_edges', 'mask_nodes'],
            33: ['subgraph', 'permute_edges', 'mask_nodes'],
            40: ['drop_nodes', 'subgraph', 'permute_edges', 'mask_nodes']
        }

    def forward(self, data):
        rand_num = self.num // 10
        ri = np.random.randint(rand_num)
        chosen_augmentations = self.num_to_augmentations.get(self.num, [])

        aug_func = self.augmentations[chosen_augmentations[ri]]
        data = aug_func(data, self.aug_ratio)
        
        return data

    def acr(self):
        return f"R{self.num}"            


