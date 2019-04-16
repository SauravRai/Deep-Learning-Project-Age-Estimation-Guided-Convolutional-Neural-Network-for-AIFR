"""
@author: Saurav Rai
This is the Pytorch implementation of the paper
Age Estimation Guided Convolutional Neural Network for Age-Invariant Face 
Recognition, Tianyue Zheng, Weihong Deng, Jiani Hu , 2017, CVPR on MORPH Dataset
"""
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

import time

from utils.light_cnn import LightCNN_4Layers
from utils.agedata import AgeFaceDataset
from torch.utils.data import Dataset, DataLoader
from utils import settings

from utils.ageutils import AgeEstGuidedModel, adjust_learning_rate
from utils import ageutils

def weights_init(m):
    classname = m.__class__.__name__
                               
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        
def main():
    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    

    agemodel = AgeEstGuidedModel(num_classes = settings.args.num_classes,num_age= settings.args.num_age)
    agemodel = nn.DataParallel(agemodel).to(device)

    lightcnnmodel = LightCNN_4Layers(num_classes = settings.args.num_classes)
    lightcnnmodel = nn.DataParallel(lightcnnmodel).to(device)
  
    params1 = []
    
    for name, param in lightcnnmodel.named_parameters():
        
        if 'fc2' not in name: 
            param.requires_grad = False
        else:
            params1.append(param)
        
                
    for name, param in agemodel.named_parameters():
    	params1.append(param)            
    
    optimizer = torch.optim.SGD(params1, lr = settings.args.lr, momentum = 0.9)
                                             
    if settings.args.resume is False:
       
        pre_trained_dict = torch.load('./LightenedCNN_4_torch.pth', map_location = lambda storage, loc: storage) 
        #pre_trained_dict = pre_trained_dict['state_dict']
        
        
        model_dict = lightcnnmodel.state_dict()          
         
        # THIS ONE IS FOR CHANGING THE NAME IN THE MODEL:
	    #NOTE : THAT IF WE ARE USING CUDA THEN WE NEED TO MENTION ( module ) LIKE HOW WE HAVE DONE BELOW         
        
        pre_trained_dict['module.features.0.filter.weight'] = pre_trained_dict.pop('0.weight')
        pre_trained_dict['module.features.0.filter.bias'] = pre_trained_dict.pop('0.bias')
        pre_trained_dict['module.features.2.filter.weight'] = pre_trained_dict.pop('2.weight')
        pre_trained_dict['module.features.2.filter.bias'] = pre_trained_dict.pop('2.bias')
        pre_trained_dict['module.features.4.filter.weight'] = pre_trained_dict.pop('4.weight')
        pre_trained_dict['module.features.4.filter.bias'] = pre_trained_dict.pop('4.bias')
        pre_trained_dict['module.features.6.filter.weight'] = pre_trained_dict.pop('6.weight')
        pre_trained_dict['module.features.6.filter.bias'] = pre_trained_dict.pop('6.bias')
        pre_trained_dict['module.fc1.filter.weight'] = pre_trained_dict.pop('9.1.weight')
        pre_trained_dict['module.fc1.filter.bias'] = pre_trained_dict.pop('9.1.bias')
        pre_trained_dict['module.fc2.weight'] = pre_trained_dict.pop('12.1.weight')
        pre_trained_dict['module.fc2.bias'] = pre_trained_dict.pop('12.1.bias')
        
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if ("fc2" not in k)}  

        model_dict.update(pre_trained_dict)  
        '''
        print('in dictionary ',pre_trained_dict['module.features.0.filter.bias'])
        for name, param in lightcnnmodel.named_parameters():
            #print(name)
            if name in ['module.features.0.filter.bias']:
                print('before loading', param.data)
        #print('after loop')
        '''
        lightcnnmodel.load_state_dict(model_dict, strict = False)          
        
        """for name, param in lightcnnmodel.named_parameters():
            if name in ['module.features.0.filter.bias']:
                print('After loading', param.data)"""
         
        
        
    else:
	#Loading from resume
        print('loading from resume')
        print('loading pretrained model')
        '''
        for name, param in lightcnnmodel.named_parameters():
            
            if name in ['module.features.0.filter.bias']:
                print('before loading', param.data)
        '''
        
        checkpoint = torch.load('3agemodel370_checkpoint.pth.tar')
        settings.args.start_epoch = checkpoint['epoch'] + 1
        lightcnnmodel.load_state_dict(checkpoint['state_dict'])
        dresmodel.load_state_dict(torch.load('4agemodel370_checkpoint.pth.tar')['state_dict'])  
        
        '''for param_group in optimizer.param_groups:
            print('lr before', param_group['lr'])
            param_group['lr'] = 0.001
            print('lr after', param_group['lr'])'''
        
        for state in optimizer.state.values():
            #print(state)
            for k, v in state.items():
                #print(k)
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
            
    train_transform = transforms.Compose([transforms.Resize(144), transforms.RandomCrop(128), transforms.ToTensor()])#,    
                                          #transforms.Normalize(mean = [0.5224], std = [0.1989])])
    
    valid_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])#,
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.1, patience = 5)
    
    train_loader = DataLoader(AgeFaceDataset(transform = train_transform, istrain = True, isvalid = False),
                    batch_size = settings.args.batch_size, shuffle = True,
                    num_workers = settings.args.workers, pin_memory = False)
    
    
    valid_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True),
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 

    test_loader = DataLoader(AgeFaceDataset(transform = valid_transform,  istrain = False, isvalid = False),
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
    
    trainprobe_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = True, isvalid = True),
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False)   #for train accuracy
    
    criterion = nn.CrossEntropyLoss().to(device)
    prev_acc1 = 0.
    prev_acc2 = 0.
    for epoch in range(settings.args.start_epoch, settings.args.epochs):
        
        test_acc = ageutils.mytest(test_loader, lightcnnmodel,agemodel, device)
        print('test accuracy : {:8f}'.format(test_acc))
        '''
        for param_group in optimizer.param_groups:
            print('lr before', param_group['lr'])
        '''
        if epoch > 0:
            galleryfeatures = ageutils.computegalleryfeatures(lightcnnmodel, agemodel, transform = valid_transform)
        
            accuracy, val_loss = ageutils.myvalidate(valid_loader,  lightcnnmodel, agemodel, criterion, optimizer, device,   galleryfeatures, istrain = False, isvalid = True)
                
            print('The valdiation accuracy and loss are: {:8f}, {}'.format(accuracy, val_loss))
        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr after', param_group['lr'])
        
        start = time.time()
        
        epoch_loss = ageutils.train(train_loader, lightcnnmodel, agemodel, criterion, optimizer, epoch, device)
        
        end = time.time()

        print('\n loss: {:.6f}, Epoch: {:d} Epochtime:{:2.2f}\n'.format(epoch_loss, epoch + 1, (end - start)))
        
        scheduler.step(test_acc)
            
        if epoch % 5 == 0 :    
            
            save_name = settings.args.save_path + 'agemodel' + str(epoch) + '_checkpoint.pth.tar'
            
            ageutils.save_checkpoint({'epoch': epoch,
                             'state_dict': lightcnnmodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                             '3'+ save_name)
            ageutils.save_checkpoint({'epoch': epoch,
                             'state_dict': agemodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                             '4'+ save_name) 
        
if __name__ == '__main__':
    settings.init()
    main()
