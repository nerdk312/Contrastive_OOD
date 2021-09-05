from numpy.core.fromnumeric import reshape
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
import copy
import random 


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k

from Contrastive_uncertainty.general.callbacks.one_dim_typicality_callback import Data_Augmented_Point_One_Dim_Class_Typicality_Normalised


    
# Used to see the dimensions of the marginal distribution which vary significantly with data augmentation in order to see the dimensions which change a lot 
# Examines the mean standard deviation value for the different values
class Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis(Data_Augmented_Point_One_Dim_Class_Typicality_Normalised):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)
        
        self.OOD_Datamodule.multi_transforms = self.Datamodule.multi_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        #
        self.augmentations = self.Datamodule.multi_transforms.num_augmentations

    def forward_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()

        multi_loader = self.Datamodule.multi_dataloader()
        multi_ood_loader = self.OOD_Datamodule.multi_dataloader()


        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        # obtain augmented features of the data
        features_test, labels_test = self.get_augmented_features(pl_module, multi_loader) #  shape (num aug, num_data_points, emb_dim)
        features_ood, labels_ood = self.get_augmented_features(pl_module, multi_ood_loader) 

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood))
    

    def normalise(self,ftrain, ftest,food):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        
        return ftrain, ftest, food

    # Used to calculate the eigenvalues and eigenvectors for the approach
    def get_1d_train(self, ftrain):
        # Nawid - get all the features which belong to each of the different classes
        cov = np.cov(ftrain.T, bias=True) # Cov and means part should be fine
        mean = np.mean(ftrain,axis=0,keepdims=True) # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues, eigvectors = np.linalg.eigh(cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/eigvalues # Calculates the scores of the training data for the different dimensions
        dtrain_1d_mean = np.mean(dtrain,axis=1, keepdims=True)
        dtrain_1d_std = np.std(dtrain, axis=1, keepdims=True)        

        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point

        #Get entropy based on training data (or could get entropy using the validation data)
        return mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std

    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        num_augmentations = self.augmentations
        
        # Calculating values for each member of the list
        ddata = [np.matmul(eigvectors.T,(fdata[i] - mean).T)**2/eigvalues for i in range(num_augmentations)]  # shape

        ddata = [ddata[i] - dtrain_1d_mean/(dtrain_1d_std +1e-10) for i in range(num_augmentations)]

        collated_data = np.stack(ddata) # shape (num augmentation,embdim, num_datapoints)
        data_std_1d = np.std(collated_data,axis=0) # standard deviation along the num augmentation dimension
        
        return data_std_1d

    def get_scores(self,ftrain_norm,ftest_norm,food_norm):
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain_norm)
        din_std_1d = self.get_thresholds(ftest_norm, mean, eigvalues, eigvectors,dtrain_1d_mean, dtrain_1d_std) #  std for all the augmentations for training data        
        dood_std_1d = self.get_thresholds(food_norm, mean, eigvalues, eigvectors,dtrain_1d_mean, dtrain_1d_std) #  std for all the augmentations for the OOD data

        return din_std_1d, dood_std_1d

    def datasaving(self,din_std_1d, dood_std_1d):
        din_average_std_1d = np.mean(din_std_1d,axis=1) # calculate the average standard deviation of each data point when using different augmentations : shape (emb dim)
        dood_average_std_1d = np.mean(dood_std_1d,axis=1)

        num_dimensions = len(din_average_std_1d)
        # concatenate the ID test and OOD values and then make dataframe
        collated_std_1d = np.stack((din_average_std_1d,dood_average_std_1d),axis=1)
        
        columns = ['ID standard deviation values', f'OOD {self.OOD_dataname} standard deviation values']
        indices = [f'Dimension {i}' for i in range(num_dimensions)]
        df = pd.DataFrame(collated_std_1d, columns=columns, index=indices)

        xs = list(range(num_dimensions))
        ys = [df['ID standard deviation values'].to_list(),df[f'OOD {self.OOD_dataname} standard deviation values'].to_list()]

        wandb.log({f'Data Augmentation Typicality Standard Deviation {self.OOD_dataname}': wandb.plot.line_series(
            xs = xs,
            ys = ys,
            keys =columns,
            title=f'Data Augmentation Typicality Standard Deviation {self.OOD_dataname}')})

    # Normalise the data
    def get_eval_results(self,ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din_std_1d, dood_std_1d = self.get_scores(ftrain_norm,ftest_norm,food_norm)        

        # Saves the line plot of the data
        self.datasaving(din_std_1d,dood_std_1d)
        # Need to concatenate the different tensors to make the dataframe
        

# Looks at the scores for only the Test dataset
class Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Single_Variance_Analysis(Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)

    
    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)

    def get_1d_train(self, ftrain):
        return super().get_1d_train(ftrain)
    
    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        return super().get_thresholds(fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std)


