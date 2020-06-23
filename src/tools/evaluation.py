#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:39:00 2020

@author: haoxuanwang
"""


import torch
from torch.autograd import Variable
from tqdm import tqdm


def evaluate_model(model, x, y, dataset_loader, task_id,**kwargs):
    total = 0
    correct = 0
    for images, labels in tqdm(dataset_loader, ascii=True):
        x.resize_(images.size()).copy_(images)
        y.resize_(labels.size()).copy_(labels)
        y=y-2*task_id

        with torch.no_grad():
            inputs = Variable(x.view(x.size(0), -1))  
            #原版为inputs = Variable(x.view(x.size(0), -1), volatile=True)，即不对与inputs有关的节点求导
        preds = model(inputs, **kwargs)

        _, predicted = torch.max(preds.data, 1)

        total += labels.size(0)
        correct += torch.tensor(predicted == y,dtype=float).sum()

    return 100.0 * float(correct) / float(total)