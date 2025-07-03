import dgl
import torch as trc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
import json
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import random

class buildGraph(DGLDataset):
    
    def __init__(self, indir,undersample_rate = 0.5,oversample_rate=1.5):
        self.indir= indir
        self.undersample_rate = undersample_rate 
        self.oversample_rate = oversample_rate
        super().__init__(name='buildgraph')
        


    def process(self):
        self.data_and_label = []
        

        trainlist = '/ifs/home/dongyihan/protein/EG-ARG/Protein_train_data/train.list' 
        label_dir =  '/ifs/home/dongyihan/protein/EG-ARG/Protein_train_data/label'
        
        node_feat_dir = '/ifs/home/dongyihan/protein/EG-ARG/Preprocessing1/processed_features/'
        edge_dir = '/ifs/home/dongyihan/protein/EG-ARG/Preprocessing1/distmaps/'
        node_xyz_dir = '/ifs/home/dongyihan/protein/EG-ARG/Preprocessing1/input/'
        f = open(trainlist, 'r')
        flines = f.readlines()
        f.close()
        for line in flines:
            tgt = line.split('.')[0]
            tgt = tgt.strip()

            labels = [] 
            labelf = open(label_dir + '/' + tgt + '.label')
            labeln = labelf.readlines()
            labelf.close()
            labellist = labeln[0].strip()
            for li in labellist:
                labels.append(int(li))

            featfile = np.load(node_feat_dir + tgt + '.3669new.npy')
            nodeFeats = trc.Tensor(featfile)
            

            nodesLeft = []
            nodesRight = []
            w = []
            epsilon = 1e-5  # 防止分母为0

            rrfile = open(edge_dir + tgt + '.dist', 'r')
            rrlines = rrfile.readlines()

            if len(rrlines[1:]) == 0:  # Sanity check
                continue

            for rline in rrlines[1:]:
                ni = int(rline.split()[0]) - 1
                nj = int(rline.split()[1]) - 1

                # Sanity check
                if (ni >= len(nodeFeats)) or (nj >= len(nodeFeats)):
                    continue

                d = float(rline.split()[4])  # 边的距离
                weight = np.log(abs(ni-nj) + 1) / (d + epsilon)  # 平滑权重

                # 可选：添加变换函数
                weight = np.tanh(weight)

                # 添加边和权重
                w.append([weight])
                w.append([weight])  # 双向边
                nodesLeft.append(ni)
                nodesRight.append(nj)
                nodesLeft.append(nj)
                nodesRight.append(ni)

            rrfile.close()

            # 可选：对权重进行归一化
            w = np.array(w)
            w = (w - w.min()) / (w.max() - w.min())  # 归一化
            w = w.tolist()




            # #### Create edge ####
            # nodesLeft = []
            # nodesRight = []
            # src = []
            # dst = []
            # w = []

            
            # rrfile = open(edge_dir + tgt + '.dist', 'r')
            # rrlines = rrfile.readlines()
            # w = []
            # ### Sanity checks
            # if(len(rrlines[1:]) == 0):
            #     continue
            # for rline in rrlines[1:]:
                
            #     ni = int(rline.split()[0])-1
            #     nj = int(rline.split()[1])-1

            #     #sanity check
            #     if((ni >= len(nodeFeats)) or (nj >= len(nodeFeats))):
            #         continue
            #     d = float(rline.split()[4])
            #     weight = np.log(abs(ni-nj))/d
            #     w.append([weight])
            #     w.append([weight])
            #     #making bi-directional edge
            #     nodesLeft.append(ni)
            #     nodesRight.append(nj)
            #     nodesLeft.append(nj)
            #     nodesRight.append(ni)
            # rrfile.close()
            src = nodesLeft
            dst = nodesRight
            xyz_f = open(node_xyz_dir + tgt + '.pdb')
            
            xyz_ca = [[0,0,0] for _ in range(len(nodeFeats))]
            xyz_flines = xyz_f.readlines()
            for xyzline in xyz_flines:
                if(xyzline[:4] == "ATOM" and xyzline[12:16].strip() == "CA"):
                    x = float(xyzline[30:38].strip())
                    y = float(xyzline[38:46].strip())
                    z = float(xyzline[46:54].strip())

                    res_no = int(xyzline[22:(22+4)]) - 1
                    if(res_no >= len(xyz_ca)):
                        continue
                    xyz_ca[res_no] = [x, y, z]
            xyz_f.close()
            xyz_ca = np.array((xyz_ca))
            
            
            edges = [src, dst]
            src = np.array(src)
            dst = np.array(dst)
            w = np.array(w)      
            xyz_feats = xyz_ca.astype(np.float32)
            xyz_feats = trc.Tensor(xyz_feats)
            labels = np.array(labels).astype(np.float32)
            self.labels = trc.Tensor([labels])            

            ### sanity check 
            if(len(nodeFeats) != len(xyz_feats)):
                nodeFeats = nodeFeats[:len(xyz_feats)]
                xyz_feats = xyz_feats[:len(nodeFeats)]
            if(len(nodeFeats) != len(labels)):
                continue
            self.nodeFeats = nodeFeats
            self.xyz_feats = xyz_feats
            self.edge_att = trc.LongTensor(w)
            self.edges = [trc.LongTensor(edges[0]), trc.LongTensor(edges[1])]
            self.data_and_label.append((self.nodeFeats, self.xyz_feats, self.edges, self.edge_att, self.labels))
            
            # Handle oversampling and undersampling
            # pos_samples = [item for item in self.data_and_label if item[4].sum() > 0]  # 正样本
            # neg_samples = [item for item in self.data_and_label if item[4].sum() == 0]  # 负样本
                
            # # 欠采样负样本
            # neg_samples_to_sample = min(len(neg_samples), int(len(pos_samples) * self.undersample_rate))
            # neg_samples = random.sample(neg_samples, neg_samples_to_sample)

            # # 过采样操作
            # pos_samples = random.choices(pos_samples, k=int(len(neg_samples) * self.oversample_rate))

            # # 将正负样本合并
            # all_samples = pos_samples + neg_samples
            

    def all_data(self):
        return [i for i in self.data_and_label]
            

    def __getitem__(self, i):
        return self.data_and_label[i]

    def __len__(self):
        return len(self.data_and_label)

