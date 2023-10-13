import json
import os
import random
import torch

from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx, degree

ROOT = './data'

def to_degree_features(data):
    d_list = []
    for graph in data:
        d_list.append(degree(graph.edge_index[0], num_nodes=graph.num_nodes))
    x = torch.cat(d_list).long()
    unique_degrees = torch.unique(x)
    mapper = torch.full_like(x, fill_value=1000000000)
    mapper[unique_degrees] = torch.arange(len(unique_degrees))
    x_onehot = torch.zeros(x.size(0), len(unique_degrees))
    x_onehot[torch.arange(x.size(0)), mapper[x]] = 1
    return x_onehot


def load_data(dataset, degree_x=True):
    if dataset == 'Twitter':
        dataset = 'TWITTER-Real-Graph-Partial'
    if dataset == 'Twitter_0.1':
        dataset = 'TWITTER-Real-Graph-Partial'
    if dataset == 'Twitter_0.02':
        dataset = 'TWITTER-Real-Graph-Partial'
    data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=dataset,
                     use_node_attr=False)
    data.data.edge_attr = None
    if data.num_node_features == 0:
        # print(dir(data.data))
        num_nodes_list = [g.num_nodes for g in data]
        data.slices['x'] = torch.tensor([0] + num_nodes_list).cumsum(0)
        # data.slices['x'] = torch.tensor([0] + data.data.num_nodes).cumsum(0)
        if degree_x:
            data.data.x = to_degree_features(data)
        else:
            num_all_nodes = sum(g.num_nodes for g in data)
            data.data.x = torch.ones((num_all_nodes, 1))
    
    print(data)
    
    return data

def divide_trn_test_val(trn_num, test_num, val_num, len, seed):
    indices = list(range(len))
    random.seed(seed)
    random.shuffle(indices)
    trn_idx_len = int(len * trn_num / (trn_num + test_num + val_num))
    tmp_idx_len = len - trn_idx_len
    trn_idx = indices[:trn_idx_len]
    tmp_idx = indices[trn_idx_len:]

    if val_num == 0:
        trn_idx = tmp_idx
        val_idx = []

    else :
        random.shuffle(tmp_idx)
        test_idx_len = int(tmp_idx_len * test_num / (test_num + val_num))
        val_idx_len = tmp_idx_len - test_idx_len
        test_idx = tmp_idx[:test_idx_len]
        val_idx = tmp_idx[test_idx_len:]

    print(trn_num, test_num, val_num)
    return trn_idx, test_idx, val_idx

def divide_dataset(trn_idx, test_idx, val_idx, num):
    trn_idx = torch.tensor(trn_idx)   
    trn_idx = trn_idx[torch.randperm(len(trn_idx))]
    trn_idx = trn_idx[:len(trn_idx) // num]
    
    test_idx = torch.tensor(test_idx)  
    test_idx = test_idx[torch.randperm(len(test_idx))]
    test_idx = test_idx[:len(test_idx) // num]
    
    val_idx = torch.tensor(val_idx)  
    val_idx = val_idx[torch.randperm(len(val_idx))]
    val_idx = val_idx[:len(val_idx) // num]

    return trn_idx.tolist(), test_idx.tolist(), val_idx.tolist()  


def load_data_split(dataset, degree_x=True, isValidate = True, seed=100):

    data = load_data(dataset, degree_x)
    if seed == 100:
        seed = int(random.random() * 2023)
    print("seed: " + str(seed))
    path = os.path.join(ROOT, 'splits', dataset, f'{seed}.json')
    if not os.path.exists(path):
        if isValidate: 
            trn_idx, test_idx, val_idx = divide_trn_test_val(8, 1, 1, len(data), seed)
        
        else:
            trn_idx, test_idx, val_idx = divide_trn_test_val(9, 1, 0, len(data), seed)

        # Getting 1/N of the data for training and testing
        if dataset == 'Twitter_0.1' :
            trn_idx, test_idx, val_idx = divide_dataset(trn_idx, test_idx, val_idx, 10)

        if dataset == 'Twitter_0.02' :
            trn_idx, test_idx, val_idx = divide_dataset(trn_idx, test_idx, val_idx, 50)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(training=trn_idx, test=test_idx, val=val_idx), f, indent=4)

    with open(path) as f:
        indices = json.load(f)

    trn_graphs = [data[i] for i in indices['training']]
    test_graphs = [data[i] for i in indices['test']]
    val_graphs = [data[i] for i in indices['val']]

    print(len(indices['training']), len(indices['test']), len(indices['val']))
    return trn_graphs, test_graphs, val_graphs

def is_connected(graph):
    return nx.is_connected(to_networkx(graph, to_undirected=True))


if __name__ == '__main__':
    load_data('PTC_MR')