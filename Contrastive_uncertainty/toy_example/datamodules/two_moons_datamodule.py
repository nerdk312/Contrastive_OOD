import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,  Dataset,Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

import os
from scipy.io import loadmat
from PIL import Image

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

import sklearn.datasets
import numpy as np
from math import ceil, floor


class TwoMoonsDataModule(LightningDataModule): # Data module for Two Moons dataset

    def __init__(self,batch_size=32,noise = 0.1,train_transforms = None, test_transforms = None):
        super().__init__()
        self.batch_size = batch_size
        self.noise = noise
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            self.X_train, self.y_train = sklearn.datasets.make_moons(n_samples=1500, noise=self.noise)
            self.mean, self.std = np.mean(self.X_train,axis = 0), np.std(self.X_train,axis = 0) # calculate the mean and std along a particular dimension

            self.X_train = (self.X_train - self.mean)/self.std #  Normalise the data

            self.X_val, self.y_val = sklearn.datasets.make_moons(n_samples=300, noise=self.noise)
            self.X_val = (self.X_val - self.mean)/self.std

        if stage == 'test' or stage is None:
            self.X_test, self.y_test = sklearn.datasets.make_moons(n_samples=200, noise=self.noise)
            self.X_test = (self.X_test - self.mean)/self.std


        #self.idx2class = {v: k for k, v in Dataset.class_to_idx.items()}
        self.idx2class  = {0:'0 - orange',1:'1 - blue'} # Dict for two moons

    def train_dataloader(self):
        '''returns training dataloader'''
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train),transform = self.train_transforms)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size,shuffle =True, drop_last = True,num_workers = 8)

        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.X_val).float(),torch.from_numpy(self.y_val), transform = self.test_transforms)
        val_loader = DataLoader(val_dataset,batch_size = self.X_val.shape[0], shuffle= False, drop_last = True,num_workers = 8) # Batch size is entire validataion set

        return val_loader

    def test_dataloader(self):
        '''returns test dataloader'''
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.y_test), transform = self.test_transforms)
        test_loader = DataLoader(test_dataset,batch_size = self.X_test.shape[0], shuffle= False, drop_last= True,num_workers = 8)  # Batch size is entire test set
        return test_loader