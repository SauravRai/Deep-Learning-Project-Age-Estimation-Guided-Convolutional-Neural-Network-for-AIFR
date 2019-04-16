'''
@author : Saurav Rai
This is the utility file function

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
import skimage.io as io
from matplotlib import pyplot as plt

import time

from utils.light_cnn import LightCNN_4Layers
from utils.agedata import AgeFaceDataset

from utils import settings



#CHANGED MADE BY SAURAV

class AgeEstModel(nn.Module):
        def __init__(self,num_classes = 10000 , num_age=58):
                super(AgeEstModel,self).__init__()
                self.fc2 = nn.Linear(256,256)
                self.fc = nn.Linear(256,num_age)
                self.fc3 = nn.Linear(256,256)
                self.fc4 = nn.Linear(256,num_classes)
        def forward(self,x):
             
                #print('The value of the feature vector x is ',x)
                #print('The type of the feature vector x is,',x.type())
                
                #x = x.view(-1,256) 
              
                age_features = self.fc2(x)
               
                #print('The size of the age_features is ',age_features.size())
                #print('The type of the age_features x is,',age_features.type())
                
                age_feat = self.fc(x)
            
                #print('The size of the age_feat is',age_feat.size())
                #print('The type of the age_feat is',age_feat.type())   
             
                age_factors = self.fc3(age_features)

                #print('The value of the age_factors is',age_factors)
                #print('The type of the age_factors is',age_factors.type())   

                #Here x is the total features we get from the LightCNN model
                sub_features = x - age_factors
                #print('The value of the sub_features is',sub_features)
                #print('The type of the sub_features is',sub_features.type())

                  
                #sub_features = sub_features.view(sub_features.size()[0], -1 )
               
                identity_features = self.fc4(sub_features) 
                #print('The value of the identity_features is',identity_features)
                #print('The type of the identity_features is',identity_features.type())
               
                #age_feat  = age_feat.view(age_feat.size()[0], -1 ) 
                #identity_features = identity_features.view(identity_features.size()[0],-1)               

                return age_feat , identity_features
                #CHANGED MADE BY SAURAV 
		#age_features is used for Age estimation task 
		#and identity_features is used for Face Recogntion Task

        
         
def AgeEstGuidedModel(**kwargs):
    model = AgeEstModel(**kwargs)
    return model       

def save_checkpoint(state,  filename):
    torch.save(state, filename)


def train(train_loader, lightcnnmodel, agemodel, criterion, optimizer, epoch, device):
    
    running_loss = 0.   
    
    data_size = 0
    
    '''for name, param in lightcnnmodel.named_parameters():
            print('name', name)
            if name in ['module.features.0.filter.weight','module.fc2.weight']:
                print('After loading in train', param.data)'''
    lightcnnmodel.train(True)
    for (x,age,label) in train_loader:
        
        optimizer.zero_grad()
        
        x = x.to(device)
         
        #print('The labels in the train are',label)
        #print('The ages in the train are',age)
        
        #print('The len of age is',len(age))
      
        #ADDED BY SAURAV RAI
        #CHANGING tuple into list and then integers
        age = list(age)
        for i in range(len(age)):
            age[i] = int(age[i])

            
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
       
        age = torch.tensor(age, dtype = torch.long)
        age = age.to(device)
       
        #print('The ages in the train  are',age) 
        #print('THe type of label is',type(label))        
        #print('THe type of age is',type(age))        
        
         
        feat, y = lightcnnmodel(x)
        #out = y #added by bala
        #print(feat.size())
        
        #BY SAURAV
        #HERE IN THIS CASE WE ARE INCREASING THE DIMENSION INTO 4D (BATCH_SIZE , NUM_CHANNELS, HEIGHT OF THE IMAGE , WIDTH OF THE IMAGE)
        #THIS IS NOT ACTUALLY NOT REQUIRED FOR ME 
        #feat = torch.unsqueeze(torch.unsqueeze(feat, dim = 2), dim = 3)
        
        #print(feat.size())
        
        #assert(False)
        #_ , out = agemodel(feat)   #added by dg on 27-08 
        #assert(False)
        
        
        age_feat , iden_feat  = agemodel(feat)   #added by dg on 27-08 
        #print(out, label, out.size(), label.size(), label.type())
        
        #print('THe size of age_feat is',age_feat.size())
        #print('The age_feat is',age_feat)
        #print('THe size of iden_feat is',iden_feat.size())
        #print('The iden_feat is',iden_feat)
        #rint('age_feat',age_feat)
        #rint('iden_feat',iden_feat)       

        loss_iden  = criterion(iden_feat, label)
        loss_age  = criterion(age_feat ,age)
        #print('epoch  & loss are ', epoch,loss.data.cpu().numpy())
        #assert(False)
        #TOTAL LOSS OF THE MODEL

        #print('loss_iden',loss_iden)
        #print('loss_age',loss_age)
        #loss = loss_iden
        
        loss = loss_iden + 0.5 * loss_age
        loss.backward()
        #print('The loss in the training part',loss.data.cpu().item())
        #print('The loss in the training part',loss)
        
        optimizer.step()
        
        #THIS I NEED TO ASK THE DOUBT
        running_loss += loss.item() * label.size(0)

        data_size += label.size(0)
    return running_loss / data_size
 


  
'''CHANGED MADE BY SAURAV RAI     
def accuracy(output, target, topk=(1,)):
    
    maxk = max(topk)
    
    batch_size = target.size(0)
    #print(output,target)
    _, pred = output.topk(maxk, 1, True, True)
    
    pred    = pred.t()
    
    #print(pred,target.view(1, -1).expand_as(pred))
    #time.sleep(10)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''


def computegalleryfeatures(lightcnnmodel, agemodel, transform):
    metafile = os.path.join('../DB', 'meta_data/age_identity_cnn_morph_metafile.pkl')  
    root_path = settings.args.root_path
        
    with open(metafile, 'rb') as fd:
        morph_dict = pickle.load(fd) 

    #gallery list will images of young sujects
    #while probe will contain the images of young subjects   
    gallerylist = []

    '''
    MADE AND COMMMENTED BY SAURAV RAI:
    self.labels = list(self.morph_dict.keys())
    self.images = list(self.morph_dict.values())
    
    for i in range(self.labels):
    	gallerylist.append(self.images[i][1])
    '''
    #GALLERY MEANS YOUNG AND PROBE MEANS OLD:   

    for (key,value) in morph_dict.items():
    	gallerylist.append(value[1])
        #print('The value of key is',key) : KEY VALUE 
        #print('The value 0 is',value[0]) : YOUNG AGE VALUE 
        #print('THe value 1 is',value[1]) : YOUNG IMAGE
        #print('THe value 2 is',value[2]) : OLD AGE VALUE
        #print('The value 3 is',value[3]) : OLD IMAGE

    '''    
    for i in range(0,10000):            
            self.gallerylist.append([i, self.allist[i][1]])  #all young faces  with label
    '''   
    #SO gallerylist contains images of 10000 young images
    gallerylist = gallerylist[0:10000]
    
    #print('gallery size',len(gallerylist))
    

    galleryfeatures = []
    
    with torch.no_grad():
        for i in range(0,len(gallerylist)):
            file1 = gallerylist[i].split('/')[-1]        
            path1 = os.path.join(root_path, file1)
            if os.path.exists(path1):
                x = Image.open(path1).convert('L')            
                x =  transform(x)
                #DOUBT NEED TO BE ASKED
                x = torch.unsqueeze(x, 0)
                #lightcnnmodel will give image features

                y, _ = lightcnnmodel(x)  # by DG
                #It will increased the dimension of the tensor to 4D 
                #y = torch.unsqueeze(torch.unsqueeze(y, dim = 2), dim = 3) #added by DG
                age_feat, iden_feat = agemodel(y)
                galleryfeatures.append(iden_feat) 
        
	   #print(len(galleryfeatures),galleryfeatures[0] )
    #assert(False)
    return galleryfeatures
    

        
        
'''
def validate(data_loader, dresmodel, lightcnnmodel, device, galleryfeatures, istrain = True ):
    #print('\nInside validation')
    acc = 0
    
    probefeatures = []
    
    with torch.no_grad():
        for (x, label) in data_loader:
            x = x.to(device)
            label = label.to(device)

            y,_ = lightcnnmodel(x)
            for j in range(len(y)):
                probefeatures.append(y[j])
            
    
        print('probe length',len(probefeatures))
        print('gallery size',len(galleryfeatures))

        correct = 0
        total = len(probefeatures)
        
        probe = np.array
        
        
        
        for i in range(len(probefeatures)):
            mindist = math.inf
            minindex = -1
            #print('probefeature size',probefeatures[i].size())
        
            
            for j in range(len(galleryfeatures)):
                #print('galleryfeature size',galleryfeatures[j].size())

                d = 1 - F.cosine_similarity(probefeatures[i].reshape(1,512), galleryfeatures[j].reshape(1,512))
                if d < mindist:
                    mindist = d
                    minindex = j

            print(' i matched with j', i,minindex)                    
            if i == minindex:
                correct += 1

        print('correct and total',correct,total)
        acc = correct * 100.0 /total    

    
   
        
    return acc
'''
def myvalidate(data_loader,  lightcnnmodel, agemodel, criterion, optimizer, device, galleryfeatures, istrain = False, 
               isvalid = True ):
    #print('\nInside validation')
    acc = 0
    if istrain is False and isvalid is True:
         galleryfeatures =  galleryfeatures[8000:10000]
    '''        
    if istrain is True and isvalid is True:
         galleryfeatures =  galleryfeatures[0:8000]
    '''    
    probefeatures = []
    target = []
    running_loss = 0.       
    data_size = 0
    
    lightcnnmodel.train(False)
    with torch.no_grad():
    #HERE We should note that the images are all old images with its corresponding ages 
        for (x,age,label) in data_loader:
            optimizer.zero_grad()
            x = x.to(device)
           
            age = list(age)
            for i in range(len(age)):
                age[i] = int(age[i])
           
            age = torch.tensor(age, dtype = torch.long)
            age = age.to(device)
            
            
            label = torch.tensor(label, dtype = torch.long)
            label = label.to(device)
           
            feat, _ = lightcnnmodel(x)
           
            #feat = torch.unsqueeze(torch.unsqueeze(feat, dim = 2), dim = 3)
           
            age_feat , iden_feat  = agemodel(feat) #by DG
           
            loss1  = criterion(iden_feat, label)
            loss2  = criterion(age_feat , age)
           
            #loss = criterion(feat,label)
            loss = loss1 + 0.5 * loss2
           
            running_loss += loss.item() * label.size(0)
           
            data_size += label.size(0)
            
            for j in range(iden_feat.size(0)):
                #DOUBT NEED TO BE ASKED:
                probefeatures.append(iden_feat[j,:].cpu().numpy())
                target.append(label[j].cpu().numpy())
    
        #print('probe length',len(probefeatures))
        #print('gallery size',len(galleryfeatures))
        gallery = []
        for i in range(len(galleryfeatures)):
            gallery.append(galleryfeatures[i].cpu().numpy().squeeze())
        
        gallery = np.array(gallery)
        
        
        total = len(probefeatures)
        
        probe = np.array(probefeatures)
        #print('The gallery features',gallery)
        #print('The probe features',probe)
        
        dist  = metrics.pairwise.cosine_similarity(probe, gallery)
        
        output = np.argmax(dist, axis = 1)
        
        target = np.array(target)
        correct =0

        #print(output.shape, target.shape, output[0:3000], target[0:3000])
        for i in range(len(probefeatures)):
            if(target[output[i]] == target[i]):
                correct +=1
 
        print('correct and total',correct,total)
        acc = correct * 100.0 /total    

    
   
        
    return acc, running_loss / data_size

def mytest(test_loader,lightcnnmodel,agemodel,device):
    #print('\nInside testing')
    
    acc = 0
    target =[]
    galleryfeatures = []
    probefeatures = []
    lightcnnmodel.train(False)
    with torch.no_grad():
        for (x1,age_yng,x2,age_old,label) in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            #print('The type of the young age is')
            #print(type(age_yng))
            num = np.array(list(age_yng),dtype=int)
            '''
            #print('THe young age are:',age_yng)
            age_yng= torch.from_numpy(num)
            age_yng = age_yng.to(device)
           
            print('The old age are:',num1)
            age_old= torch.tensor(age_old, dtype = torch.int)
            age_old = age_yng.to(device)
            '''
            label = torch.tensor(label, dtype = torch.long)
            label = label.to(device)

            feature1,_ = lightcnnmodel(x1)
            feature2,_ = lightcnnmodel(x2)
           
            #feature1 = torch.unsqueeze(torch.unsqueeze(feature1, dim = 2), dim = 3)
            #feature2 = torch.unsqueeze(torch.unsqueeze(feature2, dim = 2), dim = 3)
           
            age_feat, iden_feat1 = agemodel(feature1)
            age_feat, iden_feat2 = agemodel(feature2)
            
            
            for j in range(len(iden_feat1)):
                galleryfeatures.append(iden_feat1[j].cpu().numpy())
                
                probefeatures.append(iden_feat2[j].cpu().numpy())
               
                target.append(label[j].cpu().numpy())
         
        
        
        total = len(probefeatures)
        
        probe = np.array(probefeatures)
        gallery = np.array(galleryfeatures)
        #print('The probe features',probe)
        #print('The gallery features',gallery)

        
        dist  = metrics.pairwise.cosine_similarity(probe, gallery)
        #print(dist)
        output = np.argmax(dist, axis = 1)
        
        #target = np.arange(0,total)
        
        target = np.array(target)
        correct =0
        #print('output:',output)
        #print('target',target)
        #print(output.shape, target.shape, output[0:3000], target[0:3000])
        for i in range(len(probefeatures)):
            if(target[output[i]] == target[i]):
                correct +=1 
        
       
        
        print('correct and total',correct,total)
        acc = correct * 100.0 / total
        
        return acc

def adjust_learning_rate(optimizer, epoch):
    """decays learning rate very epoch exponentilly with gamma = 0.995"""
    
    #print('lr', settings.args.lr)
    #lr = settings.args.lr * (gamma ** epoch)
    '''if prev_acc2 - prev_acc1 > 0 and curr_acc - prev_acc2 > 0:
        gamma = 1.0005  
    elif prev_acc2 - curr_acc > 0.02:
        gamma = 0.995
    else:
        gamma = 1'''
    
    for param_group in optimizer.param_groups:
            #print('lr before', param_group['lr'])
            #param_group['lr'] *= (gamma ** epoch)
        if epoch > 3:
            param_group['lr'] = 0.0001
        #else:
            #param_group['lr'] -= .00006
            #print('lr after', param_group['lr'])
                
            
            
            
            
    
