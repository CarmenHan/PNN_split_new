#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:43:13 2020

@author: haoxuanwang
"""


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from src.data.utils import validation_split


class SplittedDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds,length, transform,id_list):
        self.parent_ds = parent_ds
        self.length = length
        self.transform = transform
        self.id_list=id_list
        super(SplittedDataset, self).__init__()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        self.parent_ds.transform = self.transform
        
        return self.parent_ds[self.id_list[i]]
        

def get_splitted_MNIST(path, batch_size,task_id):
    val_size = 10000
    normalization = transforms.Normalize((0.1307,), (0.3081,))
    transfrom = transforms.Compose([
        transforms.ToTensor(),
        normalization])
    
    total_train_set = MNIST(root=path, train=True, download=True, transform=transfrom)
    total_test_set = MNIST(root=path, train=False, download=True, transform=transfrom)
   
    train_id_list=[]
    x_train=[]
    y_train=[]
    test_id_list=[]
    x_test=[]
    y_test=[]
    
    for i,(x,y) in enumerate(DataLoader(total_train_set)):
        if y== 2*task_id or y == 2*task_id+1:
            train_id_list.append(i)
            x_train.append(x)
            y_train.append(y)

    train_set=SplittedDataset(total_train_set,len(y_train),transfrom,train_id_list)
    
    for i,(x,y) in enumerate(DataLoader(total_test_set)):
        if y== 2*task_id or y == 2*task_id+1:
            test_id_list.append(i)
            x_test.append(x)
            y_test.append(y)
    
    test_set=SplittedDataset(total_test_set,len(y_test),transfrom,test_id_list)

    train_set, val_set = validation_split(train_set, transfrom, transfrom, val_size=val_size)
    train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=False) if train_set is not None else None
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set is not None else None
    test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set is not None else None
    
    return train_loader, val_loader, test_loader
    
    
