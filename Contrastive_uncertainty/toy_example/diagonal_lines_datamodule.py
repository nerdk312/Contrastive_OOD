import os, numpy as np, matplotlib.pyplot as plt

import numpy as np
import random


import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split,  Dataset, Subset
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule


class DiagonalLinesDataModule(LightningDataModule): # Data module for Two Moons dataset

    def __init__(self,batch_size=32,noise_perc = 0.1):
        super().__init__()
        self.batch_size = batch_size
        self.noise_perc = noise_perc
        self.n_lines = 4
        self.ppline = 100
        self.intervals = [(0.1, 0.3), (0.35,0.55), (0.6, 0.8), (0.85, 1.05)]
    
    def setup(self):
        # First ppline (100) points are generated from the network for each of the line intervals and then 0.15 percent of those points are chosen from each interval (choosing 15 points out of 100 for each interval)
        lines = [np.stack([np.linspace(intv[0],intv[1],self.ppline), np.linspace(intv[0],intv[1],self.ppline)])[:,np.random.choice(self.ppline, int(self.ppline*self.noise_perc), replace=False)] for intv in self.intervals]
        #import ipdb; ipdb.set_trace()
        cls   = [x*np.ones(int(self.ppline*self.noise_perc)) for x in range(self.n_lines)] # Classes labels for each of the data points in lines
        
        self.data = np.concatenate(lines, axis=1).T
        self.labels = np.concatenate(cls) # class labels
        data_length = len(self.data)
        idxs  = np.random.choice(data_length, data_length,replace=False)
        # Shuffle the data before placing in different data to allow points in different datasets to be present
        self.data, self.labels =self.data[idxs], self.labels[idxs]

        self.train_data, self.train_labels = self.data[:int(0.7*data_length)], self.labels[:int(0.7*data_length)]
        self.val_data, self.val_labels = self.data[int(0.7*data_length):int(0.8*data_length)], self.labels[int(0.7*data_length):int(0.8*data_length)]
        self.test_data, self.test_labels = self.data[int(0.8*data_length):], self.labels[int(0.8*data_length):]

    def train_dataloader(self):
        '''returns training dataloader'''
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels))
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size,shuffle =True, drop_last = True,num_workers = 8)

        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels))
        val_loader = DataLoader(val_dataset,batch_size = self.batch_size, shuffle= False, drop_last = True,num_workers = 8) # Batch size is entire validataion set

        return val_loader

    def test_dataloader(self):
        '''returns test dataloader'''
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels))
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle= False, drop_last= True,num_workers = 8)# Batch size is entire test set
        return test_loader


Datamodule = DiagonalLinesDataModule(32,0.1)
Datamodule.setup()
Datamodule.train_dataloader()
Datamodule.val_dataloader()
Datamodule.test_dataloader()