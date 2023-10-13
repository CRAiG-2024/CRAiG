import argparse
import copy
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import warnings

from sklearn.metrics import accuracy_score
from torch_geometric.data import DataLoader

from augment import Augment, Identity, NodeDrop, Subgraph, EdgePert, AttrMask, RandAugment
from data import load_data_split, load_data
from models.gin import GIN

class CRAiGLoss(nn.Module):
    def __init__(self, threshold = 0.95, lambda_u = 1):
        super().__init__()
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.activate = torch.sigmoid

    def forward(self, weak, strong, label, y_label):
        loss_s = F.cross_entropy(label, y_label, reduction='mean')

        pseudo_label = self.activate(weak)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        loss_u = F.cross_entropy(strong, max_idx)
        loss_u = (loss_u * mask).mean()

        return loss_s + self.lambda_u * loss_u


class SoftCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        if target.ndim == 1:
            return self.ce_loss(input, target)
        elif input.size() == target.size():
            input = self.log_softmax(input)
            return self.kl_loss(input, target)
        else:
            raise ValueError()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['DD', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'MUTAG', 'PTC_MR'], help='dataset')

    # augmentation
    parser.add_argument('--weak-aug', choices=['node-drop', 'subgraph', 'edge-pert', 'attr-mask', 'rand-augment'], help='first augmentation type')    
    parser.add_argument('--strong-aug', choices=['node-drop', 'subgraph', 'edge-pert', 'attr-mask', 'rand-augment'], help='second augmentation type')
    parser.add_argument('--weak-ratio', choices=[0, 0.05, 0.1, 0.15, 0.2], help='first augmentation ratio')
    parser.add_argument('--strong-ratio', choices=[0, 0.05, 0.1, 0.15, 0.2], help="second augmentation ratio")
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--loss-lambda', type=float, default=1.0)
    parser.add_argument('--weak-rand-num', choices=[20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 40], help='first random augmentation pool')
    parser.add_argument('--strong-rand-num', choices=[20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 40], help='second random augmentation pool')

    # trials
    parser.add_argument('--trial', type=int, help='trial number')

    # experiment
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', choices=[32, 128])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--decay', choices=[0, 0.5])
    parser.add_argument('--device', type=int, default=1, help='gpu cuda number')
    parser.add_argument('--verbose', type=int, default=10)

    # classifier
    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--dropout', type=int, default=0)

    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()

    return args

@torch.no_grad()
def eval_acc(model, loader, device, metric='acc'):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        y_pred.append(output.argmax(dim=1).cpu())
        y_true.append(data.y.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return accuracy_score(y_true, y_pred)
 

@torch.no_grad()
def eval_loss(model, loss_func, loader, device):
    model.eval()
    count_sum, loss_sum = 0, 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_func(output, data.y).item()
        loss_sum += loss * len(data.y)
        count_sum += len(data.y)
    return loss_sum / count_sum

def divide_by_label(len):
    label_num = int(len / 10)
    rand = torch.randperm(len)
    label_idx = rand[:label_num]
    unlabel_idx = rand[label_num:]

    return label_idx, unlabel_idx

def into_batch(label_idx, unlabel_idx, batch_size):
    label_batch_num = int(batch_size / 10)
    if label_batch_num == 0:
        label_batch_num = 1
    label_batch_idx = label_idx[torch.randperm(len(label_idx))][:label_batch_num]
    unlabel_batch_idx = unlabel_idx[torch.randperm(len(unlabel_idx))][:(batch_size-label_batch_num)]
    return label_batch_idx, unlabel_batch_idx

def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')

def to_data(index, graph):
    data_list = []
    for i in index:
        data = copy.deepcopy(graph[i])
        data_list.append(data)
        del data
    return torch_geometric.data.Batch.from_data_list(data_list)

def to_augment(type, ratio, num):
    if type == 'node-drop':
        return NodeDrop(ratio)
    elif type == 'subgraph':
        return Subgraph(ratio)
    elif type == 'edge-pert':
        return EdgePert(ratio)
    elif type == 'attr-mask':
        return AttrMask(ratio)
    elif type == 'rand-augment':
        return RandAugment(num, ratio)
    else:
        raise RuntimeError()

def main(args):
    run_time = time.time()

    weak_aug = to_augment(args.weak_aug, args.weak_ratio, args.weak_rand_num)
    strong_aug = to_augment(args.strong_aug, args.strong_ratio, args.strong_rand_num)

    print(f"[{args.dataset}]{weak_aug.acr()}_{weak_aug.aug_ratio}/{strong_aug.acr()}_{strong_aug.aug_ratio} ({args.trial}) - start")
    
    device = to_device(args.device)
    print(device)

    ## load data as TUDataset Object
    data = load_data(args.dataset)
    num_features = data.num_features
    num_classes = data.num_classes

    trn_graphs, val_graphs, test_graphs = load_data_split(args.dataset, seed=args.seed)
    trn_loader = DataLoader(trn_graphs, batch_size=args.batch_size)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    num_batch = math.ceil(len(trn_graphs) / args.batch_size)

    model = GIN(num_features, num_classes, args.units, args.layers, args.dropout)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    loss_func = CRAiGLoss(threshold=args.threshold, lambda_u=args.loss_lambda)
    loss_func_test = SoftCELoss()

    augment = Augment(copy.deepcopy(trn_graphs), weak_aug, strong_aug)

    out_list = dict(trn_loss=[], trn_acc=[], test_loss=[], test_acc=[])
    total_time = time.time()
    local_time = time.time()
    label_idx, unlabel_idx = divide_by_label(len(trn_graphs))

    if args.verbose > 0:
        print("")
        print(' epochs\t   loss\ttrn_acc\tval_acc\t   time\t t_time')
    
    best_test_acc = 0.0  
    best_trn_acc = 0.0
    best_epoch = -1
    best_model_state_dict = None

    model_path = f"./result/models/{args.seed}/{args.dataset}_{weak_aug.acr()}_{weak_aug.aug_ratio}_{strong_aug.acr()}_{strong_aug.aug_ratio}_{args.trial}.pth"
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for _ in range(num_batch):
            label_batch_idx, unlabel_batch_idx = into_batch(label_idx, unlabel_idx, args.batch_size)
            label_data = to_data(label_batch_idx, trn_graphs)
            weak_data, strong_data = augment(unlabel_batch_idx)

            label_data = label_data.to(device)
            weak_data = weak_data.to(device)
            strong_data = strong_data.to(device)
            preds_label = model(label_data.x, label_data.edge_index, label_data.batch)
            preds_weak = model(weak_data.x, weak_data.edge_index, weak_data.batch)
            preds_strong = model(strong_data.x, strong_data.edge_index, strong_data.batch)
            
            loss = loss_func(preds_strong, preds_weak, preds_label, label_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        trn_loss = loss_sum / num_batch
        trn_acc = eval_acc(model, trn_loader, device)
        test_loss = eval_loss(model, loss_func_test, val_loader, device)
        test_acc = eval_acc(model, val_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_trn_acc = trn_acc
            best_epoch = epoch
            best_model_state_dict = model.state_dict().copy()

        out_list['trn_loss'].append(trn_loss)
        out_list['trn_acc'].append(trn_acc)
        out_list['test_loss'].append(test_loss)
        out_list['test_acc'].append(test_acc)
        
        if args.verbose > 0 and (epoch + 1) % args.verbose == 0:
            print(f'{epoch + 1:7d}\t{trn_loss:7.4f}\t{trn_acc:7.4f}\t{test_acc:7.4f}\t{(time.time()-local_time):7.4f}\t{(time.time()-total_time):7.4f}')
            local_time = time.time()

    torch.save(best_model_state_dict, model_path)
    print(f"Best test accuracy: {best_test_acc:.4f} achieved at epoch {best_epoch}")

    print(f"[{args.dataset}]{weak_aug.acr()}_{weak_aug.aug_ratio}/{strong_aug.acr()}_{strong_aug.aug_ratio} ({args.trial}) - end : {time.time()-run_time:.2f} sec")
    return 


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main(parse_args())