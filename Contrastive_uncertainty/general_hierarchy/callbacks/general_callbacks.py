import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import sklearn.metrics as skm

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import ModelSaving,quickloading,\
    SupConLoss, Centroid_distance


    
        
# Calculation of MMD based on the definition 2/ equation 1 in the paper        
# http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchetal12.pdf?origin=publication_detail
class MMD_distance(pl.Callback):
    def __init__(self,Datamodule,quick_callback):
        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback
        self.log_name = 'MMD_distance'
    '''
    def on_test_epoch_end(self,trainer, pl_module):
        self.calculate_MMD(pl_module)
    '''
    # Log MMD whilst the network is training
    def on_validation_epoch_end(self, trainer,pl_module):
        self.calculate_MMD(pl_module)

    def calculate_MMD(self, pl_module):
        dataloader = self.Datamodule.train_dataloader()
        with torch.no_grad():
            MMD_values = []
            low = torch.tensor(-1.0).to(device=pl_module.device)
            high = torch.tensor(1.0).to(device=pl_module.device)
            uniform_distribution = torch.distributions.uniform.Uniform(low,high) # causes all samples to be on the correct device when obtainig smaples https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
            #uniform_distribution =  torch.distributions.uniform.Uniform(-1,1).sample(output.shape)
            loader = quickloading(self.quick_callback,dataloader) # Used to get a single batch or used to get the entire dataset
            for data, target, coarse_targets, indices in loader:
                if isinstance(data, tuple) or isinstance(data, list):
                    data, *aug_data = data # Used to take into accoutn whether the data is a tuple of the different augmentations

                data = data.to(pl_module.device)
                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = pl_module.callback_vector(data) # representation of the data
                
                uniform_samples = uniform_distribution.sample(output.shape)
                uniform_samples = torch.nn.functional.normalize(uniform_samples,dim=1) # obtain normalized representaitons on a hypersphere
                # calculate the difference between the representation and samples from a unifrom distribution on a hypersphere
                diff = output - uniform_samples
                MMD_values.append(diff.cpu().numpy())


        MMD_list = np.concatenate(MMD_values)
        MMD_dist = np.mean(MMD_list)
        # Logs the MMD distance into wandb
        wandb.log({self.log_name:MMD_dist})

        return MMD_dist

class Uniformity(pl.Callback):
    def __init__(self, t,datamodule,quick_callback):
        super().__init__()
        self.t  = t
        self.datamodule = datamodule
        self.quick_callback = quick_callback
        self.log_name = 'uniformity'
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)
    '''
    
    def on_validation_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)

    def obtain_features(self,pl_module):
        features = []
        dataloader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, label,coarse_labels, indices) in enumerate(loader):
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            img = img.to(pl_module.device)
            features.append(pl_module.callback_vector(img))
        
        features = torch.cat(features) # Obtain the features for the representation
        return features

    def calculate_uniformity(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features) # Change to a torch tensor

        uniformity = torch.pdist(features, p=2).pow(2).mul(-self.t).exp().mean().log()
        wandb.log({self.log_name:uniformity.item()})
        #uniformity = uniformity.item()
        return uniformity
