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

from Contrastive_uncertainty.toy_NCA.datamodules.toy_transforms import CustomTensorDataset


class TwoMoonsDataModule(LightningDataModule): # Data module for Two Moons dataset

    def __init__(self,batch_size=32, train_transforms=None, test_transforms=None, noise=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.noise = noise
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            self.train_data, self.train_labels = sklearn.datasets.make_moons(n_samples=1000, noise=self.noise)
            self.mean, self.std = np.mean(self.train_data,axis = 0), np.std(self.train_data,axis = 0) # calculate the mean and std along a particular dimension

            self.train_data = (self.train_data - self.mean)/self.std #  Normalise the data

            self.val_data, self.val_labels = sklearn.datasets.make_moons(n_samples=600, noise=self.noise)
            self.val_data = (self.val_data - self.mean)/self.std

        if stage == 'test' or stage is None:
            self.test_data, self.test_labels = sklearn.datasets.make_moons(n_samples=600, noise=self.noise)
            self.test_data = (self.test_data - self.mean)/self.std

        self.train_dataset = CustomTensorDataset((torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)),transform = self.train_transforms)
        self.val_dataset = CustomTensorDataset((torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)),transform = self.test_transforms)
        self.test_dataset = CustomTensorDataset((torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)),transform = self.test_transforms)
       

        self.idx2class  = {0:'0 - orange',1:'1 - blue'} # Dict for two moons

    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True,num_workers = 8)

        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,num_workers = 8) # Batch size is entire validataion set

        return val_loader

    def test_dataloader(self):
        '''returns test dataloader'''
        
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)  # Batch size is entire test set
        return test_loader


