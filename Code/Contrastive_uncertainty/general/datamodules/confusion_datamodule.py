from typing import List
from pytorch_lightning.core import datamodule
from Contrastive_uncertainty.general.train.train_general import train
from pytorch_lightning.utilities import seed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,  Dataset,Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import copy

import os
from scipy.io import loadmat
from PIL import Image

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

import sklearn.datasets
import numpy as np
from math import ceil, floor


from Contrastive_uncertainty.general.datamodules.datamodule_transforms import CustomTensorDataset,dataset_with_indices, dataset_with_indices_SVHN

from Contrastive_uncertainty.general.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule

class ConfusionDatamodule(LightningDataModule):
    def __init__(self,ID_Datamodule, OOD_Datamodule, data_dir:str = None, batch_size =32):
        super().__init__()
        
        self.batch_size = batch_size
        self.ID_Datamodule = ID_Datamodule
        # Remove the coarse labels
        self.ID_Datamodule.DATASET_with_indices = dataset_with_indices(ID_Datamodule.DATASET)
        self.ID_Datamodule.setup()
        # Train and test transforms are defined in the datamodule dict 
        train_transforms = self.ID_Datamodule.train_transforms
        test_transforms = self.ID_Datamodule.test_transforms

        # Update the OOD transforms with the transforms of the ID datamodule
        self.OOD_Datamodule = OOD_Datamodule

        # Hack to make SVHN dataset work for the task
        if self.OOD_Datamodule.name =='svhn':
            self.OOD_Datamodule.DATASET_with_indices = dataset_with_indices_SVHN(OOD_Datamodule.DATASET)
        else:
            self.OOD_Datamodule.DATASET_with_indices = dataset_with_indices(OOD_Datamodule.DATASET)
        self.OOD_Datamodule.train_transforms = train_transforms
        self.OOD_Datamodule.test_transforms = test_transforms
        # Resets the OOD datamodules with the specific transforms of interest required
        self.OOD_Datamodule.setup()
        '''
        loader = self.OOD_Datamodule.train_dataloader()
        for i in loader:
            print(i)
        '''
        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            classes
        """
        return self.ID_Datamodule.num_classes + self.OOD_Datamodule.num_classes
    
    @property
    def num_coarse_classes(self):
        ''' 
        Return:
            classes//2
        '''
        return self.ID_Datamodule.num_coarse_classes

    @property
    def num_hierarchy(self):
        '''
        Return:
            number of layers in hierarchy
        '''
        return self.ID_Datamodule.num_hierarchy

    @property
    def num_channels(self):
        """
        Return:
            0
        """
        return self.ID_Datamodule.num_channels
    
    # Outputs the mapping for the coarse vector
    @property
    def coarse_mapping(self):
        """
        Return:
            mapping to coarse labels
        """
        return self.ID_Datamodule.coarse_mapping

    def setup(self):
        # Obtain the train, val and test datasets
        self.setup_train()
        self.setup_val()
        self.setup_test()
        self.setup_ood_test()

        
    def setup_train(self):
        
        self.train_dataset = self.concatenate_data(self.ID_Datamodule.train_dataset, self.OOD_Datamodule.train_dataset)

    def setup_val(self):
        if hasattr(self.ID_Datamodule,'val_test_dataset'):
            ID_dataset = self.ID_Datamodule.val_test_dataset
        else:
            ID_dataset = self.ID_Datamodule.val_dataset

        if hasattr(self.OOD_Datamodule,'val_test_dataset'):
            OOD_dataset = self.OOD_Datamodule.val_test_dataset
        else:
            OOD_dataset = self.OOD_Datamodule.val_dataset
        
        self.val_dataset = self.concatenate_data(ID_dataset, OOD_dataset)
        
    def setup_test(self):
        self.test_dataset = self.concatenate_data(self.ID_Datamodule.test_dataset,self.OOD_Datamodule.test_dataset)

    # OOD dataset with labels which are changed in order to calculate the confusion log probability
    def setup_ood_test(self):
        self.ood_dataset = copy.deepcopy(self.OOD_Datamodule.test_dataset)
        if isinstance(self.ood_dataset.targets, List):
            self.ood_dataset.targets = torch.tensor(self.ood_dataset.targets)



        self.ood_dataset.targets = self.ID_Datamodule.num_classes + self.ood_dataset.targets
        self.ood_dataset.labels = self.ood_dataset.targets 


    # Function used to combine the data
    def concatenate_data(self, ID_dataset, OOD_dataset):
        # data is a tuple for the different augmentations which are present 
        
        # Increase the value of the targets by the OOD data
        
        #OOD_dataset.targets = self.ID_Datamodule.num_classes + OOD_dataset.targets
        ID_data = copy.deepcopy(ID_dataset)
        OOD_data = copy.deepcopy(OOD_dataset)
        
        # Preprocesses the ID and OOD data so they can be concatenate
        ID_data = self.dataprocessing(ID_data)
        OOD_data = self.dataprocessing(OOD_data)
        
        if isinstance(OOD_data, Subset):
            OOD_data.dataset.targets = self.ID_Datamodule.num_classes + OOD_data.dataset.targets
            # Required for SVHN
            OOD_data.dataset.labels = OOD_data.dataset.targets

        else:
            OOD_data.targets = self.ID_Datamodule.num_classes + OOD_data.targets
            # Required for SVHN
            OOD_data.labels = OOD_data.targets
        
        '''
        if isinstance(ID_data, Subset):
            # Hack to make to a tensor
            if isinstance(ID_data.dataset.targets, List):
                ID_data.dataset.targets = torch.tensor(ID_data.dataset.targets)

        else:
            if isinstance(ID_data.targets, List):
                ID_data.targets = torch.tensor(ID_data.targets)
        
        if isinstance(OOD_data, Subset):
            # Hack to make to a tensor
            if isinstance(OOD_data.dataset.targets, List):
                OOD_data.dataset.targets = torch.tensor(OOD_data.dataset.targets)


            OOD_data.dataset.targets = self.ID_Datamodule.num_classes + OOD_data.dataset.targets
            OOD_data.dataset.labels = OOD_data.dataset.targets 
        else:
            if isinstance(OOD_data.targets, List):
                OOD_data.targets = torch.tensor(OOD_data.targets)

            OOD_data.targets = self.ID_Datamodule.num_classes + OOD_data.targets
            OOD_data.labels = OOD_data.targets
        '''

        datasets = [ID_data, OOD_data]
        concat_datasets = torch.utils.data.ConcatDataset(datasets)

        return concat_datasets
    
    # Processing before concatenation of the data    (Currently not used but can be used)
    def dataprocessing(self,data):
        if isinstance(data, Subset):
            # Hack to make to a tensor
            if isinstance(data.dataset.targets, List):
                data.dataset.targets = torch.tensor(data.dataset.targets)

        else:
            if isinstance(data.targets, List):
                data.targets = torch.tensor(data.targets)
        
        return data



    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True,num_workers = 8)
        return train_loader
    
    def deterministic_train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = True,num_workers = 8)
        
        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,num_workers = 8) # Batch size is entire validataion set
        # Change to list format a the val loader is generally used as a list
        return [val_loader]

    def test_dataloader(self):
        '''returns test dataloader'''
        
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)  # Batch size is entire test set
        return test_loader
    
    def ood_dataloader(self):
        '''returns ood dataloader'''
        
        ood_loader = DataLoader(self.ood_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)  # Batch size is entire test set
        return ood_loader
    

'''
ID_datamodule = MNISTDataModule()
OOD_datamodule = FashionMNISTDataModule()

confusion_module= ConfusionDatamodule(ID_datamodule,OOD_datamodule)
confusion_module.setup()

train_loader = confusion_module.train_dataloader()
'''