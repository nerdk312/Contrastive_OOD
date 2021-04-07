import os, numpy as np, matplotlib.pyplot as plt

import numpy as np
import random

import matplotlib.cm as cm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split,  Dataset, Subset
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from Contrastive_uncertainty.toy_example.datamodules.toy_transforms import ToyTrainDiagonalLinesTransforms, ToyEvalDiagonalLinesTransforms



class DiagonalLinesDataModule(LightningDataModule): # Data module for Two Moons dataset

    def __init__(self,batch_size=32,noise_perc = 0.9,train_transforms = None, test_transforms = None):
        super().__init__()
        self.batch_size = batch_size
        self.noise_perc = noise_perc
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.n_lines = 4
        self.subclusters = 2
        self.ppline = 100000
        #self.intervals = [(0.1, 0.3), (0.35,0.55), (0.6, 0.8), (0.85, 1.05)]
        self.intervals = self.data_creation()
        #self.intervals = [(0.1, 0.3), (0.35,0.55), (0.6, 0.8), (0.85, 1.05), (1.1, 1.3), (1.35, 1.55), (1.6, 1.8), (1.85, 2.05)]
        
    def data_creation(self):
        self.intervals = []
        for i in range(self.n_lines):
            for j in range(self.subclusters):
                #j = i
                self.intervals.append((0.1 + i + (0.3*j), 0.15 + i + (0.3*j)))
                #self.intervals.append((0.4+j, 0.6+j))

        return self.intervals

    def setup(self):
        # First ppline (100) points are generated from the network for each of the line intervals and then 0.15 percent of those points are chosen from each interval (choosing 15 points out of 100 for each interval)
        lines = [np.stack([np.linspace(intv[0],intv[1],self.ppline), np.linspace(intv[0],intv[1],self.ppline)])[:,np.random.choice(self.ppline, int(self.ppline*self.noise_perc), replace=False)] for intv in self.intervals]
        
        cls   = [x*np.ones(int(self.subclusters*self.ppline*self.noise_perc)) for x in range(self.n_lines)] # Classes labels for each of the data points in lines
        
        self.data = np.concatenate(lines, axis=1).T
        self.labels = np.concatenate(cls) # class labels
        #import ipdb; ipdb.set_trace()
        data_length = len(self.data)
        idxs  = np.random.choice(data_length, data_length,replace=False)
        # Shuffle the data before placing in different data to allow points in different datasets to be present
        self.data, self.labels =self.data[idxs], self.labels[idxs]
        
        
        self.train_data, self.train_labels = self.data[:int(0.7*data_length)], self.labels[:int(0.7*data_length)]
        #print('train data',self.train_data)
        '''
        mean  = np.mean(self.train_data,axis = 0)
        std = np.std(self.train_data,axis=0)
        print('mean',mean)
        print('std',std)
        '''                
        self.val_data, self.val_labels = self.data[int(0.4*data_length):int(0.8*data_length)], self.labels[int(0.4*data_length):int(0.8*data_length)]
        self.test_data, self.test_labels = self.data[int(0.8*data_length):], self.labels[int(0.8*data_length):]

        # Making the separate datasets for the dataloaders (made it during setup so that the test dataset can be used for the AUROC)
        self.train_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)), transform = self.train_transforms)
        self.val_dataset =  CustomTensorDataset(tensors = (torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels)), transform= self.test_transforms)
        self.test_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels)), transform = self.test_transforms)

    def visualise_data(self):
        #colors = cm.rainbow(np.linspace(0, 0.5,self.n_lines)) # Creates a list of numbers which represents colors
        #import ipdb; ipdb.set_trace()
        #colors = np.array([colors[int(sample_cls)] for sample_cls in self.labels][::-1]) # colors for train dataset
        #import ipdb; ipdb.set_trace()
        for i in range(self.n_lines):            
            loc = np.where(self.train_labels ==i)[0] # gets all the indices where the label has a certain index (this is correct I believe)
            plt.scatter(self.train_data[loc,0], self.train_data[loc,1])#, label= 'Train Cls {}'.format(i), s=40) #, color=list(colors[loc,:]), label='Train Cls {}'.format(i), s=40) # plotting the train data

        plt.savefig('practice.png')
        plt.savefig('practice.pdf')
        plt.close()
        

    def train_dataloader(self):
        '''returns training dataloader'''
        #train_dataset = CustomTensorDataset(tensors= (torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)),transform = self.train_transforms)
        train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size,shuffle =True, drop_last = True,num_workers = 8)

        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        #val_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels)),transform= self.test_transforms)
        val_loader = DataLoader(self.val_dataset,batch_size = self.batch_size, shuffle= False, drop_last = True,num_workers = 8) # Batch size is entire validataion set

        return val_loader

    def test_dataloader(self):
        '''returns test dataloader'''
        #test_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels)),transform = self.test_transforms)
        test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle= False, drop_last= True,num_workers = 8)# Batch size is entire test set
        return test_loader

# Use to apply transforms to the tensordataset  https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y, index # Added the return of index for the purpose of PCL

    def __len__(self):
        return self.tensors[0].size(0)

Datamodule = DiagonalLinesDataModule(32,0.1,train_transforms=ToyTrainDiagonalLinesTransforms(),test_transforms=ToyEvalDiagonalLinesTransforms())
Datamodule.setup()
Datamodule.visualise_data()
