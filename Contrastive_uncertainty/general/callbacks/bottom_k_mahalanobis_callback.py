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
import math
import random


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k


# Calculate the lowest values for the mahalanobis distance values
class Bottom_K_Mahalanobis(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True,
        k_values: int = 3):
        
        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.vector_level = vector_level
        self.label_level = label_level
        
        self.OOD_dataname = self.OOD_Datamodule.name
        # Number of k values to select for the loss
        self.k_values = k_values

        
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    
    # Performs all the computation in the callback
    def forward_callback(self, trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.deterministic_train_dataloader()
        
        # Use the test transform validataion loader
        
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level, self.label_level)
        
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
            

    def get_features(self, pl_module, dataloader, vector_level, label_level):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][label_level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]

            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][vector_level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

        return np.array(features), np.array(labels)

    
    def get_scores(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        din = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (ftest - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]
        
        dood = [
            np.sum(
                (food - np.mean(x, axis=0, keepdims=True))
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (food - np.mean(x, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
            for x in xc # Nawid- this calculates the score for all the OOD examples 
        ]
        # Change to numpy array
        din, dood = np.array(din), np.array(dood)

        bottom_k_din = []
        bottom_k_dood = []
        # initial values to mask , -1 chosen as this will not be the case
        kth_din_min = -1
        kth_dood_min = -1
        # iterate through the values
        masked_din = din
        masked_dood = dood 
        for k in range(self.k_values):
            
            # Change values which are equal to the lowest values to infinity
            masked_din = np.where(masked_din != kth_din_min,masked_din, math.inf)
            masked_dood = np.where(masked_dood != kth_dood_min,masked_dood, math.inf)
            
            # Obtain lowest values
            kth_bottom_din = np.min(masked_din,axis=0)
            kth_bottom_dood = np.min(masked_dood, axis=0)

            # update the lowest value for the mask
            kth_din_min = kth_bottom_din
            kth_dood_min = kth_bottom_dood
            
            # Save the lowest values present
            bottom_k_din.append(kth_bottom_din)
            bottom_k_dood.append(kth_bottom_dood)

            '''
            arrays.append(masked_din)
            print(np.count_nonzero(masked_din == math.inf))
            '''

        # Change to numpy array , shape (k_values, batch), the lowest k value is the smallest value
        bottom_k_din = np.array(bottom_k_din) 
        bottom_k_dood = np.array(bottom_k_dood)
        return bottom_k_din, bottom_k_dood 

       
    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        """
            None.
        """
        
    
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        bottom_k_din, bottom_k_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        in_columns = [f'Bottom {i+1} ID Mahalanobis Distances' for i in range(self.k_values)]
        ood_columns = [f'Bottom {i+1} OOD Mahalanobis Distances' for i in range(self.k_values)]
        
        self.data_saving(bottom_k_din, bottom_k_dood,in_columns, ood_columns, f'Bottom K Mahalanobis Distances OOD {self.OOD_dataname}')
    
    
    # Normalises the data
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

    def data_saving(self,bottom_k_din, bottom_k_dood,in_columns, ood_columns, wandb_dataname):
        bottom_k_din_df = pd.DataFrame(bottom_k_din.T)
        bottom_k_dood_df = pd.DataFrame(bottom_k_dood.T)
        k_min_df = pd.concat((bottom_k_din_df,bottom_k_dood_df),axis=1)
        #https://stackoverflow.com/questions/30647247/replace-nan-in-a-dataframe-with-random-values
        k_min_df = k_min_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        
        '''
        k_min_values = np.concatenate((bottom_k_din.T, bottom_k_dood.T),axis=1) # transpose and concatenate to get shape (batch, k din + k dood)
        k_min_df = pd.DataFrame(k_min_values)        
        '''

        
        k_min_df.columns = [*in_columns, *ood_columns]

        table_data = wandb.Table(data=k_min_df)
        wandb.log({wandb_dataname:table_data})

    
    
# Calculate the difference between the first and the second for a particular datapoint
class Bottom_K_Mahalanobis_Difference(Bottom_K_Mahalanobis):
    def __init__(self, Datamodule, OOD_Datamodule, vector_level: str, label_level: str, quick_callback: bool, k_values: int):
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback, k_values=k_values)

    def get_scores(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class

        din = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (ftest - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]


        dood = [
            np.sum(
                (food - np.mean(x, axis=0, keepdims=True))
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (food - np.mean(x, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
            for x in xc # Nawid- this calculates the score for all the OOD examples 
        ]

        # Change to numpy array
        din, dood = np.array(din), np.array(dood)


        bottom_k_din_diff = []
        bottom_k_dood_diff = []
        
        # iterate through the values
        masked_din = din
        masked_dood = dood 
        
        # initial values to mask , -1 chosen as this will not be the case
        previous_k_bottom_din = -1
        previous_k_bottom_dood = -1

        for k in range(self.k_values):
            # Change values which are equal to the lowest values to infinity
            masked_din = np.where(masked_din != previous_k_bottom_din, masked_din, math.inf)
            masked_dood = np.where(masked_dood != previous_k_bottom_dood, masked_dood, math.inf)
            
            # Obtain lowest values
            kth_bottom_din = np.min(masked_din,axis=0)
            kth_bottom_dood = np.min(masked_dood, axis=0)

            # Add the difference to the list
            if k >0:
                din_diff = kth_bottom_din - previous_k_bottom_din
                dood_diff = kth_bottom_dood - previous_k_bottom_dood
                bottom_k_din_diff.append(din_diff)
                bottom_k_dood_diff.append(dood_diff)
            
            # Update the previous lowest values with the current lowest
            previous_k_bottom_din = kth_bottom_din
            previous_k_bottom_dood = kth_bottom_dood


        # Change to numpy array , shape (k_values, batch), the lowest k value is the smallest value
        bottom_k_din_diff = np.array(bottom_k_din_diff) 
        bottom_k_dood_diff = np.array(bottom_k_dood_diff)
        return bottom_k_din_diff, bottom_k_dood_diff 


    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        
        """
            None.
        """
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        bottom_k_din_diff, bottom_k_dood_diff = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        in_columns = [f'Bottom {i+1} ID Mahalanobis Distances Differences' for i in range(self.k_values-1)]
        ood_columns = [f'Bottom {i+1} OOD Mahalanobis Distances Differences' for i in range(self.k_values-1)]

        self.data_saving(bottom_k_din_diff, bottom_k_dood_diff, in_columns, ood_columns, f'Bottom K Mahalanobis Distances Differences OOD {self.OOD_dataname}')

        

