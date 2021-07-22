from pytorch_lightning.core import datamodule
from Contrastive_uncertainty.general.train.train_general import train
from pytorch_lightning.utilities import seed
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

from Contrastive_uncertainty.toy_example.datamodules.toy_transforms import CustomTensorDataset

from Contrastive_uncertainty.toy_replica.toy_general.datamodules.blobs_datamodule import BlobsDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.two_moons_datamodule import TwoMoonsDataModule


class ConfusionDatamodule(LightningDataModule):
    def __init__(self,ID_Datamodule, OOD_Datamodule, data_dir:str = None, batch_size =32):
        super().__init__()
        
        self.batch_size = batch_size
        self.ID_Datamodule = ID_Datamodule
        self.ID_Datamodule.setup()
        # Train and test transforms are defined in the datamodule dict 
        train_transforms = self.ID_Datamodule.train_transforms
        test_transforms = self.ID_Datamodule.test_transforms

        # Update the OOD transforms with the transforms of the ID datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.train_transforms = train_transforms
        self.OOD_Datamodule.test_transforms = test_transforms
        # Resets the OOD datamodules with the specific transforms of interest required
        self.OOD_Datamodule.setup()
        
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
        OOD_data, *OOD_labels, OOD_indices = self.OOD_Datamodule.test_dataset[:]
        if isinstance(OOD_labels, tuple) or isinstance(OOD_labels, list):
            OOD_labels, *_ = OOD_labels

        # Combines the data from the different approaches present 
        
        OOD_labels = self.ID_Datamodule.num_classes + OOD_labels
        ood_dataset = [OOD_data[i] for i in range(len(OOD_data))] + [OOD_labels]
        
        self.ood_dataset = CustomTensorDataset(tuple(ood_dataset))


    # Function used to combine the data
    def concatenate_data(self, ID_dataset, OOD_dataset):
        # data is a tuple for the different augmentations which are present 
        ID_data, *ID_labels, ID_indices = ID_dataset[:]
        if isinstance(ID_labels, tuple) or isinstance(ID_labels, list):
            ID_labels, *_ = ID_labels
        
        OOD_data, *OOD_labels, OOD_indices = OOD_dataset[:]
        if isinstance(OOD_labels, tuple) or isinstance(OOD_labels, list):
            OOD_labels, *_ = OOD_labels

        # Combines the data from the different approaches present 
        #confusion_data =tuple(map(torch.cat, zip(*data)))

        
        if isinstance(ID_data,tuple) or isinstance(ID_data, list):
            confusion_data = tuple([torch.cat((ID_data[i],OOD_data[i])) for i in range(len(ID_data))])
        else:
            confusion_data = torch.cat((ID_data, OOD_data))
        
        #max_ID_label = max(ID_labels)
        OOD_labels = self.ID_Datamodule.num_classes + OOD_labels
        #import ipdb; ipdb.set_trace()
        #OOD_labels = max_ID_label +1 + OOD_labels # Need to add since it starts at zero
        confusion_labels = torch.cat((ID_labels ,OOD_labels))
        
        # confusion_dataset , made from the different values in the confusion data list as well as the confusion labels
        confusion_dataset = [confusion_data[i] for i in range(len(confusion_data))] + [confusion_labels]
        
        dataset = CustomTensorDataset(tuple(confusion_dataset))
        
        return dataset

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
ID_datamodule = BlobsDataModule()
OOD_datamodule = TwoMoonsDataModule()

#ID_datamodule.setup()
#OOD_datamodule.setup()

confusion_module= ConfusionDatamodule(ID_datamodule,OOD_datamodule)
confusion_module.setup()
'''