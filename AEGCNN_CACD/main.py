"""
@author: Saurav Rai
This is the Pytorch implementation of the paper
Age Estimation Guided Convolutional Neural Network for Age-Invariant Face 
Recognition, Tianyue Zheng, Weihong Deng, Jiani Hu , 2017, CVPR on CACD Dataset
"""
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

import time

from utils.light_cnn import LightCNN_4Layers
#from utils.light_cnn import LightCNN_29Layers //can use any of the pretrained lightcnn models
from utils.agedata import AgeFaceDataset
from torch.utils.data import Dataset, DataLoader
from utils import settings

from utils.ageutils import AgeEstGuidedModel, adjust_learning_rate

from utils import ageutils

        
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
        pre_trained_dict = torch.load('./LightenedCNN_4_torch.pth', map_location = lambda storage, loc: storage) #by DG
        #pre_trained_dict = torch.load('./LightCNN_9Layers_checkpoint.pth.tar', map_location = lambda storage, loc: storage) #by DG
        #pre_trained_dict = torch.load('./LightCNN_29Layers_V2_checkpoint.pth.tar', map_location = lambda storage, loc: storage) #by DG
        #pre_trained_dict = torch.load('./LightCNN_29Layers_checkpoint.pth.tar', map_location = lambda storage, loc: storage) #by DG
        
        #pre_trained_dict = pre_trained_dict['state_dict']
        model_dict = lightcnnmodel.state_dict()  
        
        # THIS ONE IS FOR CHANGING NAMES IN THE MODEL:
	    # IF WE ARE USING CUDA THEN WE NEED TO MENTION ( module ) LIKE HOW I HAVE DONE BELOW         
        
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
        
       # 1. filter out unnecessary keys  
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if ("fc2" not in k)}  
       # 2. overwrite entries in the existing state dict
        model_dict.update(pre_trained_dict)  
       # 3. load the new state dict  
        lightcnnmodel.load_state_dict(model_dict, strict = False)           
        
    else:
	#Loading from resume
        print('loading from resume')
        print('loading pretrained model')
        checkpoint = torch.load('3agemodel370_checkpoint.pth.tar')
        settings.args.start_epoch = checkpoint['epoch'] + 1
        lightcnnmodel.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        agemodel.load_state_dict(torch.load('4agemodel370_checkpoint.pth.tar')['state_dict'])  
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
      
    train_transform = transforms.Compose([transforms.Resize(144), transforms.RandomCrop(128), transforms.ToTensor()])#,    
    
    valid_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])    
    #TRAIN LOADER
    train_loader = DataLoader(AgeFaceDataset(transform = train_transform, istrain = True, isvalid = False , isquery = False ,
                    isgall1 = False ,isgall2 = False ,isgall3 =False),
                    batch_size = settings.args.batch_size, shuffle = True,
                    num_workers = settings.args.workers, pin_memory = False)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='max',factor =0.1, patience= 5)
    #VALID LOADER
    
    valid_query_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = True,
                    isgall1 = False , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
    
    valid_gall1_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = True , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
  
    valid_gall2_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = False , isgall2 = True ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
   
    valid_gall3_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = False , isgall2 = False ,isgall3 = True), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
    
    #TEST LOADER
    test_query_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = True,
                    isgall1 = False , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 

    test_gall1_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = False,
                    isgall1 = True , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False)
    
    test_gall2_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = False,
                    isgall1 = False , isgall2 = True ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False)
     
    test_gall3_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False ,isquery = False,
                    isgall1 = False , isgall2 = False ,isgall3 = True), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 

    
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(settings.args.start_epoch, settings.args.epochs):
        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr after', param_group['lr'])	        
        
        test_mean_avg_prec1 = ageutils.mytest_gall(test_query_loader,test_gall1_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery1 : {:8f}'.format(test_mean_avg_prec1))
        test_mean_avg_prec2 = ageutils.mytest_gall(test_query_loader,test_gall2_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery2 : {:8f}'.format(test_mean_avg_prec2))
        test_mean_avg_prec3 = ageutils.mytest_gall(test_query_loader,test_gall3_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery3 : {:8f}'.format(test_mean_avg_prec3))
        
        
        valid_mean_avg_prec1 = ageutils.mytest_gall(valid_query_loader,valid_gall1_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery1(valid) : {:8f}'.format(valid_mean_avg_prec1))
        valid_mean_avg_prec2 = ageutils.mytest_gall(valid_query_loader,valid_gall2_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery2(valid) : {:8f}'.format(valid_mean_avg_prec2))
        valid_mean_avg_prec3 = ageutils.mytest_gall(valid_query_loader,valid_gall3_loader, lightcnnmodel,agemodel, device)
        print('test mean average precision for gallery3(valid) : {:8f}'.format(valid_mean_avg_prec3))
        
        start = time.time()
        
        epoch_loss = ageutils.train(train_loader, lightcnnmodel, agemodel, criterion, optimizer, epoch, device)
        
        end = time.time()
        print('\n loss: {:.6f}, Epoch: {:d} Epochtime:{:2.2f}\n'.format(epoch_loss, epoch + 1, (end - start)))
        scheduler.step(test_mean_avg_prec1)        
        
        '''
        if epoch % 5 == 0 :    
            
            save_name = settings.args.save_path + 'agemodel' + str(epoch) + '_checkpoint.pth.tar'
            
            #ageutils.save_checkpoint({'epoch': epoch + 1,
                             #'state_dict': dresmodel.state_dict()}, '1'+ save_name)
            #ageutils.save_checkpoint({'epoch': epoch ,
                             #'state_dict': lightcnnmodel.state_dict()}, '3'+ save_name) # commented by bala on 22-7-18
            # following line added by bala on 22-07-2018 
            
            ageutils.save_checkpoint({'epoch': epoch,
                             'state_dict': lightcnnmodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                             '3'+ save_name)
            ageutils.save_checkpoint({'epoch': epoch,
                             'state_dict': agemodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                             '4'+ save_name) #by DG
            
                        
        #accuracy = ageutils.mytest(test_loader, dresmodel, lightcnnmodel, device)
        #print('test accuracy is :', accuracy)
        '''
if __name__ == '__main__':
    settings.init()
    main()
