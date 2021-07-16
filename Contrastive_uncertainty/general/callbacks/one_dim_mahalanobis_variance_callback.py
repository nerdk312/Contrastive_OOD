
from numpy.lib.financial import _ipmt_dispatcher
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
import pandas as pd
import wandb
import sklearn.metrics as skm
import faiss
import statistics 

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score


from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, get_fpr, get_pr_sklearn, get_roc_plot, get_roc_sklearn, table_saving
from Contrastive_uncertainty.general.callbacks.hierarchical_ood import kde_plot, count_plot
from Contrastive_uncertainty.general.callbacks.one_dim_mahalanobis_callback import One_Dim_Mahalanobis, One_Dim_Relative_Mahalanobis, Class_One_Dim_Relative_Mahalanobis



# Calculates variance along a dimension rather than the mean
class One_Dim_Mahalanobis_Variance(One_Dim_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)
    
    def get_scores(self, ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        #import ipdb; ipdb.set_trace()
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)
                
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        din = np.min(din,axis=0) # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        din = np.var(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.var(dood, axis=1) # Find the mean of all the data points, dood per dimension
        
        return din, dood
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        #import ipdb; ipdb.set_trace()
        xs = np.arange(len(dtest))
        #baseline = np.zeros_like(dtest)
        ys = [dtest, dood]

        # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        wandb.log({f"1D Mahalanobis Variance {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Mahalanobis Variance per dim", f"{self.OOD_dataname} OOD data Mahalanobis Variance per dim"],
                       title= f"1-Dimensional Mahalanobis Distance Variances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood

        
# Get calculates the variance instead of calculating the mean
class One_Dim_Relative_Mahalanobis_Variance(One_Dim_Relative_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)
    
    def get_scores(self, ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        #import ipdb; ipdb.set_trace()
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)
        
        ####### Specific to relative mahalanobis #############
        background_cov = np.cov(ftrain.T, bias=True)
        background_mean = np.mean(ftrain,axis=0,keepdims=True)
        background_eigvals, background_eigvectors = np.linalg.eigh(background_cov)
        background_eigvals =  np.expand_dims(background_eigvals,axis=1)
        #import ipdb; ipdb.set_trace()
        background_din = np.abs(np.matmul(background_eigvectors.T,(ftest - background_mean).T)**2/background_eigvals)
        background_dood = np.abs(np.matmul(background_eigvectors.T,(food - background_mean).T)**2/background_eigvals)

        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num]) - background_din for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        din = np.min(din,axis=0) # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        din = np.var(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) - background_dood for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.var(dood, axis=1) # Find the mean of all the data points, dood per dimension
        
        return din, dood
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        #import ipdb; ipdb.set_trace()
        xs = np.arange(len(dtest))
        #baseline = np.zeros_like(dtest)
        ys = [dtest, dood]

        # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        wandb.log({f"1D Relative Mahalanobis Variance {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Relative Mahalanobis Variance per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis Variance per dim"],
                       title= f"1-Dimensional Relative Mahalanobis Distance Variances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood




# Makes predictions using the normal mahalanobis distance but then shows the scores taking into account the background statistics   
class Class_One_Dim_Relative_Mahalanobis_Variance(Class_One_Dim_Relative_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback)

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def get_scores(self, ftrain, ftest, food, ypred, indices_test, indices_ood):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)
        

        ####### Specific to relative mahalanobis #############
        background_cov = np.cov(ftrain.T, bias=True)
        background_mean = np.mean(ftrain,axis=0,keepdims=True)
        background_eigvals, background_eigvectors = np.linalg.eigh(background_cov)
        background_eigvals =  np.expand_dims(background_eigvals,axis=1)
        
        background_din = np.abs(np.matmul(background_eigvectors.T,(ftest - background_mean).T)**2/background_eigvals)
        background_dood = np.abs(np.matmul(background_eigvectors.T,(food - background_mean).T)**2/background_eigvals)

        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # CALCULATES DIN AND DOOD using the relative mahalanobis scores
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num]) - background_din for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) - background_dood for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        
        # Go through the different classes in the true labels
        collated_din_class = []
        collated_dood_class = []
        for i in np.unique(ypred):
            din_class = din[i].T # change from shape (Embdim, B) to shape (B, embemdim)
            din_class = din_class[indices_test==i] # obtain all the indices which are predicted as this class , shape (class_batch, embdim)
            din_class = np.var(din_class,axis=0) # Mean of all the data points in the class
            
            dood_class = dood[i].T
            dood_class = dood_class[indices_ood ==i]
            dood_class = np.var(dood_class,axis=0)
                
            collated_din_class.append(din_class)
            collated_dood_class.append(dood_class)
        
        # Change to values of -1 to represent it is not a number as the variance needs to be positive in a typical case
        collated_din_class = np.nan_to_num(collated_din_class,nan=-1.0)
        collated_dood_class = np.nan_to_num(collated_dood_class,nan =-1.0)
        return collated_din_class, collated_dood_class

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        indices_dtest, indices_dood = self.get_predictions(ftrain_norm, ftest_norm, food_norm, labelstrain)
        collated_class_dtest, collated_class_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, indices_dtest, indices_dood)
        # Plot the ID and OOD data on separate graphs as there are too many values for a specific class
        xs = np.arange(len(collated_class_dtest[0]))
        ID_keys = [f'ID data Class {class_num}' for class_num in np.unique(labelstrain)]        

        wandb.log({f"ID Class Wise 1D Relative Mahalanobis Variances" : wandb.plot.line_series(
                       xs=xs,
                       ys=collated_class_dtest,
                       keys = ID_keys,
                       # keys= ["ID data Relative Mahalanobis per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis per dim"],
                       title= f"ID data Class Wise 1-Dimensional Relative Mahalanobis Distance Variances",
                       xname= "Dimension")})
        
        OOD_keys = [f"OOD Class {class_num}" for class_num in np.unique(labelstrain)]

        wandb.log({f"{self.OOD_dataname} OOD Class Wise 1D Relative Mahalanobis Variances" : wandb.plot.line_series(
                       xs=xs,
                       ys=collated_class_dood,
                       keys = OOD_keys,
                       # keys= ["ID data Relative Mahalanobis per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis per dim"],
                       title= f"{self.OOD_dataname} OOD data Class Wise 1-Dimensional Relative Mahalanobis Distance Variances",
                       xname= "Dimension")})


        '''
        collated_ys = []
        collated_keys = []
        
        
        for class_num in np.unique(labelstrain):
            #xs = np.arange(len(collated_class_dtest[class_num]))

            y_id, y_ood = collated_class_dtest[class_num], collated_class_dood[class_num]
            collated_ys.append(y_id)
            collated_ys.append(y_ood)

            key_id, key_ood = f"ID data Class {class_num} Relative Mahalanobis Variance per dim", f"{self.OOD_dataname} OOD data Class {class_num} Relative Mahalanobis per dim"
            collated_keys.append(key_id)
            collated_keys.append(key_ood)
            
            #ys = [collated_class_dtest[class_num],collated_class_dood[class_num]]
            #keys= [f"ID data Class {class_num} Relative Mahalanobis Variance per dim", f"{self.OOD_dataname} OOD data Class {class_num} Relative Mahalanobis per dim"]

            # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        
        
        wandb.log({f"Class Wise 1D Relative Mahalanobis Variances {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=collated_ys,
                       keys = collated_keys,
                       # keys= ["ID data Relative Mahalanobis per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis per dim"],
                       title= f"Class Wise 1-Dimensional Relative Mahalanobis Distance Variances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})
        '''
        return collated_class_dtest, collated_class_dood