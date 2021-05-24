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

class GaussianBlobs(LightningDataModule): # Data module for Two Moons dataset
    def __init__(self, batch_size=32, train_transforms=None, test_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_datapoints = 10000
        self.num_classes = 12
        self.visualise_name = 'gaussian_blobs'

    def setup(self):    
        # Make the datasets for the case where there is different values
        self.train_data, self.train_labels = self.data_creation(int(0.7*self.num_datapoints))
        self.mean, self.std = np.mean(self.train_data,axis = 0), np.std(self.train_data,axis = 0)
        self.train_data = (self.train_data - self.mean)/self.std 

        self.val_data, self.val_labels = self.data_creation(int(0.1*self.num_datapoints))
        self.val_data = (self.val_data - self.mean)/self.std 

        self.test_data, self.test_labels = self.data_creation(int(0.2*self.num_datapoints))
        self.test_data = (self.test_data - self.mean)/self.std 

        # Making the separate datasets for the dataloaders (made it during setup so that the test dataset can be used for the AUROC)
        self.train_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)), transform=self.train_transforms)
        self.val_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels)), transform=self.test_transforms)
        self.test_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels)), transform=self.test_transforms)

    def visualise_data(self):
        for i in range(self.num_classes):           
            loc = np.where(self.train_labels ==i)[0] # gets all the indices where the label has a certain index (this is correct I believe)
            plt.scatter(self.train_data[loc,0], self.train_data[loc,1])#, label= 'Train Cls {}'.format(i), s=40) #, color=list(colors[loc,:]), label='Train Cls {}'.format(i), s=40) # plotting the train data

        
        plt.savefig('Images/' + self.visualise_name + '.png')
        plt.savefig('Images/' + self.visualise_name + '.pdf')
        plt.close()
        
        
    def data_creation(self, num_datapoints):
        data = []
        cov = [[1,0],[0,1]]
        # calculate different mean each time to get different samples for the data
        for i in range(self.num_classes):
            mean = [20*np.cos(i*np.pi/(2*np.pi)),20*np.sin(i*np.pi/(2*np.pi))]
            class_data = np.random.multivariate_normal(mean, cov,num_datapoints)
            data.append(class_data)
        # calculate labels for the different class data
        labels = [x*np.ones(int(num_datapoints)) for x in range(self.num_classes)]
        # change the list format to a numpy array format
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels)

        data_length = len(data)
        idxs  = np.random.choice(data_length, data_length, replace=False)
        # Shuffle the data before placing in different data to allow points in different datasets to be present
        data, labels = data[idxs], labels[idxs]

        return data, labels

    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size,shuffle =True, drop_last = True,num_workers = 8)
        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        val_loader = DataLoader(self.val_dataset,batch_size = self.batch_size, shuffle= False, drop_last = True,num_workers = 8) # Batch size is entire validataion set

        return val_loader

    def test_dataloader(self):
        '''returns test dataloader'''
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


# Only difference is that the code for the val dataset is different
class PCLGaussianBlobs(GaussianBlobs):
    def __init__(self, batch_size=32, train_transforms=None, test_transforms=None):
        super(PCLDiagonalLines, self).__init__(batch_size, train_transforms, test_transforms) 
    
    # updates the val dataset to be the same as the train dataset but uses a different augmentation for the particular task
    def setup(self):
        # First ppline (100) points are generated from the network for each of the line intervals and then 0.15 percent of those points are chosen from each interval (choosing 15 points out of 100 for each interval)
        lines = [np.stack([np.linspace(intv[0],intv[1],self.ppline), np.linspace(intv[0],intv[1],self.ppline)])[:,np.random.choice(self.ppline, int(self.ppline*self.noise_perc), replace=False)] for intv in self.intervals]
        
        cls   = [x*np.ones(int(self.subclusters*self.ppline*self.noise_perc)) for x in range(self.n_lines)] # Classes labels for each of the data points in lines
        
        self.data = np.concatenate(lines, axis=1).T
        self.labels = np.concatenate(cls) # class labels
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
        #self.val_data, self.val_labels = self.data[int(0.7*data_length):int(0.8*data_length)], self.labels[int(0.7*data_length):int(0.8*data_length)]
        self.test_data, self.test_labels = self.data[int(0.8*data_length):], self.labels[int(0.8*data_length):]

        # Val dataset is the same as the train dataset except that the transformation is different
        self.train_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)), transform = self.train_transforms)
        self.val_dataset =  CustomTensorDataset(tensors = (torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)), transform= self.test_transforms)
        self.test_dataset = CustomTensorDataset(tensors = (torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels)), transform = self.test_transforms)
    
    

Datamodule = GaussianBlobs(32)
Datamodule.setup()
Datamodule.visualise_data()
