'''
@author: Saurav Rai
This is the settings file
'''
import argparse
import torch

parser = argparse.ArgumentParser(description = 'AIFR')

parser.add_argument('-j', '--workers', default = 1, type = int, metavar = 'N',
                    help = 'number of data loading workers (default: 4)')

parser.add_argument('--epochs', default = 100, type = int, metavar = 'N', help = 'number of total epochs to run')

parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N', help = 'manual epoch number')

parser.add_argument('-b', '--batch_size', default = 64, type = int,
                    metavar = 'N', help = 'mini-batch size (default: 64)')

parser.add_argument('--lr', '--learning-rate', default = 1e-3, type = float,
                    metavar = 'LR', help = 'initial learning rate')

parser.add_argument('--momentum', default = 0.99, type = float, metavar = 'M',  help = 'momentum')

parser.add_argument('--weight_decay', '--wd', default = 0.001, type = float,
                    metavar = 'W', help ='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default = 100, type = int,
                    metavar = 'N', help = 'print frequency (default: 100)')

parser.add_argument('--num_classes', '-c', default = 1520, type = int,
                    metavar = 'N', help = 'Number of Identity classes for FR (default: 1520) Tot:1520')

parser.add_argument('--num_age', '-age', default = 49, type = int,
                    metavar = 'N', help = 'Number of Age classes for FR (default: 49) Tot:49')

parser.add_argument('--root_path', default = '', type = str, metavar = 'PATH',
                    help = 'path to root path of images (default: none)')


parser.add_argument('--save_path', default = '', type = str, metavar = 'PATH',
                    help = 'path to save checkpoint (default: none)')

parser.add_argument('--resume', default = False, type = bool, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')

parser.add_argument('--resumefile', default = 'agemodel21_checkpoint.pth', type = str, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')



def init():
    global args
    global device
    args = parser.parse_args('--root_path /data/Saurav/DB/CACD2000Cropped/'.split()) #Set the path of the CACDCropped dataset 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
