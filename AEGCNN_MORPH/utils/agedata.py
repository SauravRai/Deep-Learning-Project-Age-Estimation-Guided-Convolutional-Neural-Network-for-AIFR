'''
This is the agedata file for the MORPH Dataset
'''
from torch.utils.data import Dataset
from utils import settings
import os
import scipy.io as sio
import pickle
import numpy as np
from sklearn import metrics
from PIL import Image
import torch

class AgeFaceDataset(Dataset):
    def __init__(self, transform = None, istrain = False, isvalid = False):
        super().__init__()
       
        self.transform = transform
        
        self.istrain = istrain
        self.isvalid = isvalid
                        
        self.metafile = os.path.join('../DB', 'meta_data/age_identity_cnn_morph_metafile.pkl')  
        self.root_path = settings.args.root_path
        
        with open(self.metafile, 'rb') as fd:
            self.morph_dict = pickle.load(fd)  
        
        self.labels = list(self.morph_dict.keys())
        self.images = list(self.morph_dict.values())
        
        #CHANGES MADE BY SAURAV RAI : 
        self.allist = []
        self.trainlist = []
        self.validlist = []
        #validlist = []
        
        
        
        self.testgallerylist =  []
        self.testprobelist =  []
        
        self.trainprobelist =  []
        
        self.young_age = []
        self.young_image = []
        self.old_age = []
        self.old_image = []

        print('The no of classes is',len(self.labels)) 
        assert(len(self.labels) == len(self.images))
        
        '''   
        MADE AND CHANGED BY SAURAV RAI 
        for i in range(self.labels):
	    self.young_age.append(self.images[i][0])
            self.young_image.append(self.images[i][1])
            self.old_age.append(self.images[i][2])
            self.old_image.append(self.images[i][3])
        '''
       
        '''        
        print('The young image age',self.images[1][0])
        print('The young images are:',self.images[1][1])
        print('The old image age is:',self.images[1][2])  
        print('The old images are:',self.images[1][3])        
        '''
        dic ={}
        
        self.list_train_ages =[]

        for i in range(0,10000):
            if(self.images[i][2] not in self.list_train_ages):
                self.list_train_ages.append(self.images[i][2])

        for i in range(0,10000):
            if(self.images[i][0] not in self.list_train_ages):
                self.list_train_ages.append(self.images[i][0])
        print('The length of the list_train_ages:',len(self.list_train_ages))
        for i in range(len(self.list_train_ages)):
            dic.update({self.list_train_ages[i]:i})
     

    
        for i in range(8000, 10000):#len(self.labels)):
            self.validlist.append([i, self.images[i][3],self.images[i][2]])  #labels , old  faces , old ages  

        for i in range(0,8000):
            self.trainprobelist.append([i, self.images[i][3],self.images[i][2]])  #old faces  with label
        
        
        for i in range(0,10000):
            self.trainlist.append([i, self.images[i][3],dic[self.images[i][2]]])  #old faces  with label
            '''           
            print('THe oldimage ',self.trainlist[i][1])
            print('THe label ',self.trainlist[i][0])
            print('The age of the old image is',self.trainlist[i][2])
            '''
                    
        for i in range(0,10000):
            self.trainlist.append([i, self.images[i][1],dic[self.images[i][0]]])  #all young faces  with label
           
        '''
            print('THe youngimage ',self.trainlist[i][1])
            print('THe label ',self.trainlist[i][0])
            print('The age of the young image is',self.trainlist[i][2])
        '''
            #self.trainlist.append([i, self.images[i][0]])  #all young ages  with label
            #print('THe youngimage',self.trainlist[i][2])
            #print('THe youngage',self.trainlist[i][1])
        
        
        for i in range(10000, 13000): #len(self.labels)):
            self.testgallerylist.append([i, self.images[i][1],self.images[i][0]]) # labels , young test faces , young ages 
            #self.testgallerylist.append([i, self.images[i][0]]) #young age test  
            self.testprobelist.append([i, self.images[i][3],self.images[i][2]])  #labels , old test faces , old ages
            #self.testprobelist.append([i, self.images[i][2]])  #old age faces 
        ''' 
        for i in range(0,1):
            print('THE VALUE OF trainlist[i][0]',self.trainlist[i][0])       
            print('THE VALUE OF trainlist[i][1]',self.trainlist[i][1])       
            print('THE VALUE OF trainlist[i][1]',self.trainlist[i][2])       
     
        '''
        #print('The value of the trainlist in 0 is:',self.trainlist[19999])

    def __len__(self):
        
        if self.istrain is True  and self.isvalid is False:
            return len(self.trainlist)
        
        if self.istrain is False  and self.isvalid is True:
             
            return len(self.validlist)
        
        if self.istrain is False and self.isvalid is False: 
            return len(self.testprobelist)
        
        if self.istrain is True  and self.isvalid is True: 
            return len(self.trainprobelist)
        
    def __getitem__(self, i):
        
        if self.istrain is True  and self.isvalid is False:     #Training set old and young
            
            #This will contain the image part
            file1 = self.trainlist[i][1].split('/')[-1]
            #print('The file in train:',file1)
       	    #This will contain the age part 
            age_part = self.trainlist[i][2]
            #print('The age_part in train:',age_part)
            label = self.trainlist[i][0]
            #print('The label in train:',label)
            path1 = os.path.join(self.root_path, file1)
        
            #print(path1,label)
                 
            if os.path.exists(path1):
                x = Image.open(path1).convert('L')

                if self.transform:
                    x =  self.transform(x)

                return x,age_part,label
            
        if self.istrain is False  and self.isvalid is True:   #validation set old 
            
            #This will contain the image part 
            file1 = self.validlist[i][1].split('/')[-1]
            #This will contain the age of the image
            age_part = self.validlist[i][2]
            label = self.validlist[i][0]-8000
            
             
            path1 = os.path.join(self.root_path, file1)
        
            #print(path1,label)
                 
            if os.path.exists(path1):
                x = Image.open(path1).convert('L')

                if self.transform:
                    x =  self.transform(x)


                return x,age_part,label
            
        if self.istrain is False and self.isvalid is False:    #Test set both young and old
            gallery  = self.testgallerylist[i][1].split('/')[-1]
            gallery_age =  self.testgallerylist[i][2]
            probe  = self.testprobelist[i][1].split('/')[-1]
            probe_age  = self.testprobelist[i][2]
           
            label = self.testprobelist[i][0] - 10000
            
            path1 = os.path.join(self.root_path, gallery)
            path2 = os.path.join(self.root_path, probe)
            #print(path1,label)
                 
            if os.path.exists(path1) and os.path.exists(path2):
                x = Image.open(path1).convert('L')
                y = Image.open(path2).convert('L')
                if self.transform:
                    x =  self.transform(x)
                    y =  self.transform(y)

                return x,gallery_age,y,probe_age,label
            
            
        
        if self.istrain is True  and self.isvalid is True:  #train set only old
            probe  = self.trainprobelist[i][1].split('/')[-1]
            probe_age  = self.trainprobelist[i][2]
            label = self.trainprobelist[i][0]
            
            
            path2 = os.path.join(self.root_path, probe)
            #print(path1,label)
                 
            if  os.path.exists(path2):
                y = Image.open(path2).convert('L')
                if self.transform:
                    y =  self.transform(y)
                return y,probe_age,label
        
        
       


 
