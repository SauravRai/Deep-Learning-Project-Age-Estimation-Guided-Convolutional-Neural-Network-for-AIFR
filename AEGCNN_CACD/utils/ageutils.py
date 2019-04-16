'''
 This file contains the architecture proposed in the paper and severl other utility functions
'''
from torch.utils.data import Dataset
from utils import settings
import os
import scipy.io as sio
import pickle
import numpy as np
from sklearn import metrics
from PIL import Image
import torchvision.transforms as transforms


import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import math

import time
from utils.agedata import AgeFaceDataset

from utils import settings

#CHANGED MADE BY SAURAV

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class AgeEstModel(nn.Module):
        def __init__(self,num_classes = 1520 , num_age=49):
                super(AgeEstModel,self).__init__()
                self.feat = nn.Sequential(
                            mfm(256,256,type =0))
                self.fc2 = nn.Linear(256,256)
                self.fc = nn.Linear(256,num_age)
                self.fc3 = nn.Linear(256,256)
                self.fc4 = nn.Linear(256,num_classes)
        def forward(self,x):
                print('The x shape from lightcnn model:',x.size())
                x = self.feat(x)
                age_features = self.fc2(x)
                
                age_feat = self.fc(x)
                act2 = self.feat(age_features)  
                age_factors = self.fc3(act2)
                
                sub_features = x - age_factors
                act3 = self.feat(sub_features)
                identity_features = self.fc4(act3) 

                return age_feat , identity_features

        
         
def AgeEstGuidedModel(**kwargs):
    model = AgeEstModel(**kwargs)
    return model       

def save_checkpoint(state,  filename):
    torch.save(state, filename)


def train(train_loader, lightcnnmodel, agemodel, criterion, optimizer, epoch, device):
    
    running_loss = 0.   
    
    data_size = 0
    
    lightcnnmodel.train(True)
    for (x,age,label) in train_loader:
        
        optimizer.zero_grad()
        
        x = x.to(device)
        
        age =list(age)
        for i in range(len(age)):
            age[i] = int(age[i])
         
            
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
       
        age = torch.tensor(age, dtype = torch.long)
        age = age.to(device)
        #chanegs made due to the lcnn architecture
        feat, y = lightcnnmodel(x)
        #y , feat = lightcnnmodel(x)
        
        age_feat , iden_feat  = agemodel(feat)   #added by dg on 27-08 
        
        loss_iden  = criterion(iden_feat, label)
        loss_age  = criterion(age_feat ,age)

        loss = loss_iden +  loss_age
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * label.size(0)
        data_size += label.size(0)
    
    return running_loss / data_size

def mytest_gall(test_query_loader,test_gall_loader ,lightcnnmodel,agemodel,device):
    
    acc = 0
    target =[]
    target2 =[]
    query_features = []
    gallery_features = []
    q_features = []
    g_features = []
    lightcnnmodel.train(False) 


    with torch.no_grad():
        for (x1,age1,label1) in test_query_loader:
            x1 = x1.to(device)
            feature1,_ = lightcnnmodel(x1)
            #_ ,feature1 = lightcnnmodel(x1)
            age_feat , iden_feat = agemodel(feature1)
            query_features = iden_feat
                        
            label1 = torch.tensor(label1, dtype = torch.long)
            label1 = label1.to(device)
            
            for j in range(len(query_features)):
           
                q_features.append(query_features[j].cpu().numpy())
                target.append(label1[j].cpu().numpy())       
        for (x2 ,age2,label2) in test_gall_loader:
            x2 = x2.to(device)
            feature2,_ = lightcnnmodel(x2)
            #_ ,feature2 = lightcnnmodel(x2)
            age_feat , iden_feat = agemodel(feature2)
            gallery_features = iden_feat

            label2 = torch.tensor(label2, dtype = torch.long)
            label2 = label2.to(device)
            
            for j in range(len(gallery_features)):
               
                g_features.append(gallery_features[j].cpu().numpy())
                target2.append(label2[j].cpu().numpy())       
                
         
        total = len(q_features)
        
        label1 = np.array(label1)
        label2 = np.array(label2)

        q_features = np.array(q_features)
        g_features = np.array(g_features)
        #print('The query features',q_features)
        
        #print('The gallery features',g_features)
        target = np.array(target)
        target2 = np.array(target2)
        
        
        dist  = metrics.pairwise.cosine_similarity(q_features, g_features)
        #print('THE SHAPE OF dist matrix is',dist.size)
        Avg_Prec =0
        Total_average_precision = 0
        
        #WE WILL BE SELECTING THE 5 CLOSEST FEATURES 
        k=5

        for i in range(len(q_features)):
            correct_count = 0
            prec = 0
            
            idx = np.argpartition(dist[i],-k)
            indices = idx[-k:] #THIS WILL GIVE ME THE INDICES OF 5 HIGHEST VALUE         
            true_label  = target[i]
            
          
            for j in range(1,len(indices)+1):
                #print('The target2 values are:',target2[indices[j-1]])
                if(true_label == target2[indices[j-1]]):
                    correct_count = correct_count + 1
                    prec = prec + float(correct_count) / j
                if(correct_count == 0):
                    correct_count =1
                    prec = 0
            Avg_Prec = Avg_Prec + 1.0/correct_count * prec
       
        Total_average_precision =  Avg_Prec
       
        Mean_Average_Precision = 1/total * Total_average_precision
        
    return Mean_Average_Precision    
   

def adjust_learning_rate(optimizer, epoch):
    
    for param_group in optimizer.param_groups:
        if epoch > 3:
            param_group['lr'] = 0.0001
                
            
            
            
            
    
