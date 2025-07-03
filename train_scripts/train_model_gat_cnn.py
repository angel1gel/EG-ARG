import random
from egat_clean_dr01_cnn import *
import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dgl.dataloading import GraphDataLoader
import dgl
import math
import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_dataloader import buildGraph
from Dataloader import buildGraph as test_buildGraph
from GetLabel import get_labels
import pickle


from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
import numpy as np
import torch

# 计算FPR
def calculate_fpr(tp, tn, fp, fn):
    return fp / (fp + tn) if (fp + tn) != 0 else 0

# 计算Precision, Recall, F1, MCC
def calculate_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    return precision, recall, f1, mcc

def test_epoch(model, dataloader, PARS):
    model.eval()

    names = []
    preds = []

    rloss = 0
    for i, (data_feats) in enumerate(dataloader):
        (tgt_name, nodeFeats, xyz_feats, edges, edge_att) = data_feats
        tgt_name = tgt_name[0]
        names.append(tgt_name)
        print('running ' + tgt_name + ' ...')
        
        n_nodes = len(nodeFeats[0])
        n_e = len(edges[0])
        nodeFeats = nodeFeats.to(PARS.device)
        xyz_feats = xyz_feats.to(PARS.device)
        edges[0] = edges[0].to(PARS.device)
        edges[1] = edges[1].to(PARS.device)
        edge_att = edge_att.to(PARS.device)

        nodeFeats = nodeFeats.squeeze()
        xyz_feats = xyz_feats.squeeze()
        edges[0] = edges[0].squeeze()
        edges[1] = edges[1].squeeze()
        edge_att = edge_att.squeeze()
        edge_att = edge_att.unsqueeze(dim=1)

        pred, xyz = model(nodeFeats, xyz_feats, edges, edge_att)
        pred = torch.nn.Sigmoid()(pred)
        # pred = pred.detach().cpu().numpy()
        pred = np.squeeze(pred.detach().cpu().numpy()).tolist()

        #print(pred)
        fo = open('/ifs/home/dongyihan/protein/EG-ARG/output/' + tgt_name + '.out', 'w')
        for pr in pred:
            fo.write(str(pr) + '\n')
        fo.close()
        
        preds += pred

    labels = get_labels('/ifs/home/dongyihan/protein/EG-ARG/Protein_train_data/label', names)
    results = preds
    # 这里是保存的路径
    # 保存为 .pkl 文件
    with open('/ifs/home/dongyihan/protein/EG-ARG/svm_pred/gat_EquiPNAS_pred_.pkl', 'wb') as f:
        pickle.dump(np.array(results), f)
    with open('/ifs/home/dongyihan/protein/EG-ARG/svm_pred/gat_EquiPNAS_labels_.pkl', 'wb') as f:
        pickle.dump(np.array(labels), f)
 
    # 通过调整阈值来控制FPR
    # 设定一个阈值范围来控制假阳性率
    threshold = 0.5  # 你可以调整这个值来控制 FPR
    
    preds_bin = [1 if pred >= threshold else 0 for pred in preds]

    # 计算TP, TN, FP, FN
    tp = np.sum((np.array(preds_bin) == 1) & (np.array(labels) == 1))
    tn = np.sum((np.array(preds_bin) == 0) & (np.array(labels) == 0))
    fp = np.sum((np.array(preds_bin) == 1) & (np.array(labels) == 0))
    fn = np.sum((np.array(preds_bin) == 0) & (np.array(labels) == 1))


    # 计算FPR
    fpr = calculate_fpr(tp, tn, fp, fn)

    # 计算Precision, Recall, F1, MCC
    precision, recall, f1, mcc = calculate_metrics(tp, tn, fp, fn)

    # 计算其他常用指标
    acc = accuracy_score(labels, preds_bin)
    roc_auc = roc_auc_score(labels, results)

    fpr = round(fpr, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    mcc = round(mcc, 4)
    acc = round(acc, 4)
    roc_auc = round(roc_auc, 4)
    
    # 返回保持不变的值
    return roc_auc, acc, recall, precision, f1, mcc


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss



def to_np(x):
    return x.cpu().detach().numpy()



def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS, test_loader):
    
    scheduler.step()
    print('epoch ' + str(epoch))
    best_auc = 0
    early_stop = 0

    # # 在训练之前检查一个batch的数据
    # for i, data_and_label in enumerate(dataloader):
    #     print(f"Data and Label at batch {i}: {data_and_label}")
    #     break  # 只检查第一个batch


    for i, data_and_label in enumerate(dataloader):
        
        (nodeFeats, xyz_feats, edges, edge_att, y) = data_and_label
        if len(data_and_label) == 5:
            nodeFeats, xyz_feats, edges, edge_att, y = data_and_label
            
        else:
            print(f"Unexpected data format in data_and_label. Length: {len(data_and_label)}")
        
       
        
        n_nodes = len(nodeFeats[0])
        n_e = len(edges[0])
        nodeFeats = nodeFeats.to(FLAGS.device)
        xyz_feats = xyz_feats.to(FLAGS.device)
        edges[0] = edges[0].to(FLAGS.device)
        edges[1] = edges[1].to(FLAGS.device)
        edge_att = edge_att.to(FLAGS.device)
        y = y.to(FLAGS.device)
        model.train()
        optimizer.zero_grad()
        
        nodeFeats = nodeFeats.squeeze()
        xyz_feats = xyz_feats.squeeze()
        edges[0] = edges[0].squeeze()
        edges[1] = edges[1].squeeze() 
        edge_att = edge_att.squeeze()
        edge_att = edge_att.unsqueeze(dim=1)
        y = y.squeeze()
        y = y.unsqueeze(dim=1)

        # 使用 torch.any 判断是否包含正样本
        if torch.any(y == 1):  # 判断是否包含正样本
            nodeFeats, edges, edge_att = augment_graph(nodeFeats, edges, edge_att)
            
        pred, xyz = model(nodeFeats, xyz_feats, edges, edge_att)  # Pass xyz_feats as x
        
        l1_loss = loss_fnc(pred, y)
        l1_loss.backward()
        optimizer.step()

    all_result = test_epoch(model, test_loader, FLAGS)

    return all_result

        

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)

def main(FLAGS, UNPARSED_ARGV):

    # Data 
    # dataset = buildGraph(FLAGS.indir)
    dataset = buildGraph(FLAGS.indir, undersample_rate=0.3, oversample_rate=1.5)
    train_loader = GraphDataLoader(dataset, batch_size=1, shuffle=True)
    test_loader = GraphDataLoader(test_buildGraph(FLAGS.testdir), batch_size=1, shuffle=False)

    FLAGS.train_size = len(train_loader)

    model = EGATWithCNN(in_node_nf=3669, hidden_nf=FLAGS.hidden_nf, out_node_nf=1, in_edge_nf=1, 
             n_layers=FLAGS.num_layers, device=FLAGS.device)
   
    #print(model)
    model.to(FLAGS.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 80) # #of epochs 80
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name +  '_bce-EG-ARG.pt')

    # Train
    best_auc = 0
    early_stop = 0
    
    print('Begin training')
    for epoch in range(80): # #of epochs 50
        #print(f"Saved: {save_path}")
        task_loss = torch.nn.BCELoss() 

        # # 假设我们有一个二分类问题，正样本的权重设置为4
        # pos_weight = torch.tensor([60]).to(FLAGS.device)
        pos_weight = torch.tensor([25]).to(FLAGS.device)
        # 定义损失函数
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # task_loss = FocalLoss()
        all_results = train_epoch(epoch, model, criterion, train_loader, optimizer, scheduler, FLAGS, test_loader)
        if all_results[0] > best_auc:
            best_auc = all_results[0]
            temp = all_results
            torch.save(model.state_dict(), save_path)
            early_stop = 0
            print(f"-------------best_auc:{best_auc}--------------------")
            print(f"-------------temp:{temp}--------------------")
        else:
            early_stop += 1
        if early_stop > 10:
            print(f"-------------early_stop:{temp}--------------------")
            break
    print(f"-------------best:{temp}--------------------")
    model.load_state_dict(torch.load(save_path), strict=False)
    test_epoch(model, test_loader, FLAGS)
        # torch.save(model.state_dict(), save_path)
    print(f"Training done. Model saved in: {save_path}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import random
def augment_graph(node_feats, edges, edge_att, augment_factor=0.1):
    # 将 edge_att 转换为 float 类型
    edge_att = edge_att.float()

    # 增强 edge_att，添加一些噪声
    augmented_edge_att = edge_att + torch.randn_like(edge_att) * augment_factor

    # 你可以根据需要继续对 node_feats 或 edges 做增强
    return node_feats, edges, augmented_edge_att



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--num_layers', type=int, default=2,
            help="Number of equivariant layers")
    parser.add_argument('--hidden_nf', type=int, default=128,
            help="Number of hidden nf")
    parser.add_argument('--indir', type=str, default="/ifs/home/dongyihan/protein/EG-ARG/Protein_train_data",
            help="Input data directory")
    parser.add_argument('--save_dir', type=str, default="models/",
            help="Directory name to save models")
    parser.add_argument('--testdir', type=str, default='/ifs/home/dongyihan/protein/EG-ARG/Preprocessing1',
            help="Path to input data containing distance maps and input features")
    parser.add_argument('--seed', type=int, default=42)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Name
    FLAGS.name = f'E-l{FLAGS.num_layers}-{FLAGS.hidden_nf}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)

    FLAGS.device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    main(FLAGS, UNPARSED_ARGV)
