from numpy.core.defchararray import lower, upper
from numpy.lib.function_base import quantile
from pandas.io.formats.format import DataFrameFormatter
import torch
from torch._C import dtype
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sklearn.metrics as skm
import faiss
import statistics

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k


class Total_Centroid_KL(pl.Callback):
    def __init__(self, Datamodule, quick_callback:bool = True):
        super().__init__()

        self.Datamodule = Datamodule
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)
    
    def forward_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        #train_loader = self.Datamodule.train_dataloader()


        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(labels_train))
    
    def get_features(self, pl_module, dataloader):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = 0
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]

            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = pl_module.callback_vector(img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())            

        return np.array(features), np.array(labels)


    # Normalises the data
    def normalise(self,ftrain):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10

        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)

    
        return ftrain
   
    def get_scores(self, ftrain, ypred): 
        #https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence 
        #https://pytorch.org/docs/stable/distributions.html
        # Need to also have enough data present otherwise the covariance matrix will be invertible 
        # Need to make sure that the mean is one dimensional , rather than 2 dimensional 

        # Need to train Ftrain into the correct shape
        # Need to also have enough data present otherwise the covariance matrix will be invertible

        ftrain = ftrain.astype(np.float64)
        total_cov = torch.tensor(np.cov(ftrain.T, bias=True))
        total_mean = torch.tensor(np.mean(ftrain,axis=0)) # Need to make sure that the mean has a single dimension when forming the covariance matrix
        
        
        total_multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(total_mean,total_cov)
        '''
        m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros
        (2), torch.eye(2)
        '''

        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class

        # Calculate gaussians for the individual class
        class_covs = [torch.tensor(np.cov(x.T, bias=True)) for x in xc] # Cov and means part should be fine
        class_means = [torch.tensor(np.mean(x,axis=0)) for x in xc] # Calculates mean from (B,embdim) to (emb_dim) need to make sure that I do not have 2 dimensions to use it for the KL divergence situation

        class_multivariate_normals = [torch.distributions.multivariate_normal.MultivariateNormal(class_means[class_num],class_covs[class_num]) for class_num in range(len(class_covs))]
        KLs = [torch.distributions.kl.kl_divergence(total_multivariate_normal,class_multivariate_normals[class_num]) for class_num in range(len(class_multivariate_normals))]
        columns = ['KL(nats)']
        indices = [f'Class {i}' for i in range(len(class_multivariate_normals))]
        df = pd.DataFrame(KLs,index=indices, columns=columns)
        table = wandb.Table(dataframe=df)
        wandb.log({'KL Divergence(Total||Class)':table})
        # Plot the KL divergences in a table
        return KLs
    
    def get_eval_results(self, ftrain, labelstrain):
        ftrain_norm = self.normalise(ftrain)
        KLs = self.get_scores(ftrain_norm, labelstrain)
        return KLs 


# Calculates the overlap between the data points in different classes
class Class_Centroid_Radii_Overlap(Total_Centroid_KL):
    def __init__(self, Datamodule, quick_callback: bool, lower_percentile:int = 75, upper_percentile:int = 95):
        super().__init__(Datamodule, quick_callback=quick_callback)

        self.lower_percentile= lower_percentile
        self.upper_percentile= upper_percentile

    # calculate centroids
    def get_cluster_centroids(self, ftrain, ypred):
        centroids = []
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        centroids = np.array([np.mean(x, axis=0) for x in xc])
        return centroids


    def get_radii(self, ftrain, ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        class_centroids = [np.mean(x,axis=0,keepdims=True) for x in xc]
        diff = [xc[class_num] - class_centroids[class_num] for class_num in range(len(class_centroids))]
        radii = [np.linalg.norm(class_data,axis=1) for class_data in diff] # Changes shape from (class_batch, dim) to (class_batch)
        
        #class_centroids = np.array(np.squeeze(class_centroids)) # need to squeeze to get shape (num classes, emb dim) otherwise shape will be (num classes, 1, embdim)
        return class_centroids,radii

    def get_scores(self, ftrain, ypred): 
        
        class_centroids, radii = self.get_radii(ftrain, ypred)
        lower_radii = [np.percentile(class_radii,self.lower_percentile) for class_radii in radii]
        upper_radii = [np.percentile(class_radii,self.upper_percentile) for class_radii in radii]
        #lower_radii = [round(np.percentile(class_radii,self.lower_percentile),2) for class_radii in radii]
        #upper_radii = [round(np.percentile(class_radii,self.upper_percentile),2) for class_radii in radii]
        
        # obtain all datapoints besides the data point from interest
        xc = [ftrain[ypred != i] for i in np.unique(ypred)] 

        diff = [xc[class_num]- class_centroids[class_num] for class_num in range(len(class_centroids))]
        # calculates the radii between the class centroids and points not in the class
        non_class_radii = [np.linalg.norm(non_class_data,axis=1) for non_class_data in diff]
        
        # Check values in the non class which is lower than the lower radii for the class
        lower_radii_present = [non_class_radii[class_num][non_class_radii[class_num] <lower_radii[class_num]] for class_num in range(len(class_centroids))]
        lower_radii_present = [len(lower_radii_present[class_num])/len(non_class_radii[class_num]) for class_num in range(len(class_centroids))]
        lower_radii_present = np.around(np.array(lower_radii_present), decimals=3)

        upper_radii_present = [non_class_radii[class_num][non_class_radii[class_num] <upper_radii[class_num]] for class_num in range(len(class_centroids))]
        upper_radii_present = [len(upper_radii_present[class_num])/len(non_class_radii[class_num]) for class_num in range(len(class_centroids))]
        upper_radii_present = np.around(np.array(upper_radii_present), decimals=3)

        collated_radii_present= np.stack((lower_radii_present,upper_radii_present),axis=1)

        columns = [f'Fraction present - {self.lower_percentile}th percentile class radius', f'Fraction present - {self.upper_percentile}th percentile class radius']
        indices = [f'Class {i}' for i in range(len(xc))]
        df = pd.DataFrame(collated_radii_present,index=indices, columns=columns)
        table = wandb.Table(dataframe=df)
        wandb.log({'Non Class Percentage Overlap':table})
        # Plot the KL divergences in a table
        
        return df
        
        
    
    def get_eval_results(self, ftrain, labelstrain):
        ftrain_norm = self.normalise(ftrain)
        df = self.get_scores(ftrain_norm,labelstrain)
        return  df