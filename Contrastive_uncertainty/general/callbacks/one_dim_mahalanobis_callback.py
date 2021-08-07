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
import scipy

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, get_fpr, get_pr_sklearn, get_roc_plot, get_roc_sklearn, table_saving
from Contrastive_uncertainty.general.callbacks.hierarchical_ood import kde_plot, count_plot

# Callback to calculate scores for a particular level, not for the hierarchical case
class One_Dim_Mahalanobis(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level, label_level, quick_callback)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        pass
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self, trainer, pl_module):
        #print('General scores being used') 
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level, self.label_level)

        dtest, dood = self.get_eval_results(
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
    
    def get_scores(self, ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            #eigvals, eigvectors = np.linalg.eigh(class_cov) #scipy.linalg.eigh(class_cov)
            

            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals, axis=1))
            eigvectors.append(class_eigvectors)
        
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        # eigenvalues has shape  (128,1) whilst the matrix multiplication term as shape (128,B), therefore the eigenvaleus are broadcast to 128,B
        din = np.min(din,axis=0) # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        din = np.mean(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.mean(dood, axis=1) # Find the mean of all the data points, dood per dimension
        
        '''
        array = []
        array1 = np.array([[3,4],[1,2],[5,6],[0,9]])
        array2 = np.array([[1,2],[3,4],[5,6],[7,8]])
        array3 = np.array([[3,5],[1,3],[4,6],[1,7]])
        array.append(array1)
        array.append(array2)
        array.append(array3)
        '''
        
        return din, dood

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        xs = np.arange(len(dtest))
        #baseline = np.zeros_like(dtest)
        ys = [dtest, dood]

        # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        wandb.log({f"1D Mahalanobis {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Mahalanobis per dim", f"{self.OOD_dataname} OOD data Mahalanobis per dim"],
                       title= f"1-Dimensional Mahalanobis Distances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood

# Get scores function uses the background statistics in the results also
class One_Dim_Relative_Mahalanobis(One_Dim_Mahalanobis):
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
        din = np.mean(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) - background_dood for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.mean(dood, axis=1) # Find the mean of all the data points, dood per dimension
        
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
        wandb.log({f"1D Relative Mahalanobis {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Relative Mahalanobis per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis per dim"],
                       title= f"1-Dimensional Relative Mahalanobis Distances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood


# Get scores function uses the background statistics in the results also
class One_Dim_Background_Mahalanobis(One_Dim_Mahalanobis):
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
        
        # Find the mean of the background statistics for all the data points
        background_din = np.mean(background_din,axis=1)
        background_dood = np.mean(background_dood,axis =1)
               
        return background_din, background_dood
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        background_dtest, background_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        #import ipdb; ipdb.set_trace()
        xs = np.arange(len(background_dtest))
        #baseline = np.zeros_like(dtest)
        ys = [background_dtest, background_dood]

        # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        wandb.log({f"1D Background Mahalanobis {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Background Mahalanobis per dim", f"{self.OOD_dataname} OOD data Background Mahalanobis per dim"],
                       title= f"1-Dimensional Background Mahalanobis Distances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return background_dtest, background_dood

# get scores function uses a shared covariance matrix 
class One_Dim_Shared_Mahalanobis(One_Dim_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback)
    
    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def get_scores(self, ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        

        #cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        # Calculation as specified in pseudocoode A for the shared covariance matrix https://arxiv.org/pdf/2106.09022.pdf
        shared_cov = [np.matmul((xc[class_num]- means[class_num]).T, (xc[class_num]- means[class_num])) for class_num in range(len(means))]
        # Elementwise sum of the covariance matrices in the list followed by division by all the elements present 
        shared_cov = np.sum(shared_cov,axis=0)/(len(ftrain))

        #shared_cov[0][0,1] + shared_cov[1][0,1] + shared_cov[2][0,1] + shared_cov[3][0,1] + shared_cov[4][0,1]+shared_cov[5][0,1]+shared_cov[6][0,1]+shared_cov[7][0,1]+shared_cov[8][0,1] +shared_cov[9][0,1]
        # Output should be 128 by 128

        eigvalues, eigvectors = np.linalg.eigh(shared_cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        # import ipdb; ipdb.set_trace()
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors.T,(ftest - means[class_num]).T)**2/eigvalues) for class_num in range(len(means))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        din = np.min(din,axis=0) # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        din = np.mean(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors.T,(food - means[class_num]).T)**2/eigvalues) for class_num in range(len(means))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.mean(dood, axis=1) # Find the mean of all the data points, dood per dimension
        
        
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
        wandb.log({f"1D Mahalanobis Shared Covariance {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Mahalanobis Shared Covariance per dim", f"{self.OOD_dataname} OOD data Mahalanobis Shared Covariance per dim"],
                       title= f"1-Dimensional Mahalanobis Shared Covariance Distances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood
# get scores function uses a background information as well as uses a shared covariance matrix in the data
class One_Dim_Shared_Relative_Mahalanobis(One_Dim_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback)
    
    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def get_scores(self, ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        

        #cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        # Calculation as specified in pseudocoode A for the shared covariance matrix https://arxiv.org/pdf/2106.09022.pdf
        shared_cov = [np.matmul((xc[class_num]- means[class_num]).T, (xc[class_num]- means[class_num])) for class_num in range(len(means))]
        # Elementwise sum of the covariance matrices in the list followed by division by all the elements present 
        shared_cov = np.sum(shared_cov,axis=0)/(len(ftrain))

        #shared_cov[0][0,1] + shared_cov[1][0,1] + shared_cov[2][0,1] + shared_cov[3][0,1] + shared_cov[4][0,1]+shared_cov[5][0,1]+shared_cov[6][0,1]+shared_cov[7][0,1]+shared_cov[8][0,1] +shared_cov[9][0,1]
        # Output should be 128 by 128

        eigvalues, eigvectors = np.linalg.eigh(shared_cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)


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
        # import ipdb; ipdb.set_trace()
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors.T,(ftest - means[class_num]).T)**2/eigvalues) - background_din for class_num in range(len(means))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        din = np.min(din,axis=0) # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        din = np.mean(din, axis=1) # Find the mean of all the data points for that particular dimension, din per dimension 

        dood = [np.abs(np.matmul(eigvectors.T,(food - means[class_num]).T)**2/eigvalues) - background_dood for class_num in range(len(means))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        dood = np.min(dood,axis=0) # Find min along the class dimension
        dood = np.mean(dood, axis=1) # Find the mean of all the data points, dood per dimension
                
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
        wandb.log({f"1D Relative Mahalanobis Shared Covariance {self.OOD_dataname}" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys,
                       keys= ["ID data Relative Mahalanobis Shared Covariance per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis Shared Covariance per dim"],
                       title= f"1-Dimensional Relative Mahalanobis Shared Covariance Distances - {self.OOD_dataname} OOD data",
                       xname= "Dimension")})

        return dtest, dood

# Calculates the 1D mahalanobis distance for the different classes
class Class_One_Dim_Mahalanobis(One_Dim_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback)

    def on_test_epoch_end(self, trainer, pl_module):
        # Only perform classwise calculation when there is less than 10 classes in the dataset
        if self.Datamodule.num_classes > 10:
            pass
        else:
            self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def get_predictions(self,ftrain, ftest, food, ypred):
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
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        return indices_din, indices_dood

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
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
       
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point
        din = [np.abs(np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present 
        dood = [np.abs(np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num]) for class_num in range(len(cov))] # Perform the absolute value to prevent issues with the absolute mahalanobis distance being present
        
        # Go through the different classes in the true labels
        collated_din_class = []
        collated_dood_class = []
        for i in np.unique(ypred):
            din_class = din[i].T # change from shape (Embdim, B) to shape (B, embemdim)
            din_class = din_class[indices_test==i] # obtain all the indices which are predicted as this class , shape (class_batch, embdim)
            din_class = np.mean(din_class,axis=0) # Mean of all the data points in the class
            
            dood_class = dood[i].T
            dood_class = dood_class[indices_ood ==i]
            dood_class = np.mean(dood_class,axis=0)
            
            
            collated_din_class.append(din_class)
            collated_dood_class.append(dood_class)
        

        # Change to values of 0 as this would not have any diffe
        collated_din_class = np.nan_to_num(collated_din_class,nan=0.0)
        collated_dood_class = np.nan_to_num(collated_dood_class,nan =0.0)
        return collated_din_class, collated_dood_class

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        indices_dtest, indices_dood = self.get_predictions(ftrain_norm, ftest_norm, food_norm, labelstrain)
        collated_class_dtest, collated_class_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, indices_dtest, indices_dood)
        
        #import ipdb; ipdb.set_trace()
        #xs = np.arange(len(dtest))
        #baseline = np.zeros_like(dtest)
        
        for class_num in np.unique(labelstrain):
            xs = np.arange(len(collated_class_dtest[class_num]))
            ys = [collated_class_dtest[class_num],collated_class_dood[class_num]]
            # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
            wandb.log({f"Class {class_num} 1D Mahalanobis {self.OOD_dataname}" : wandb.plot.line_series(
                           xs=xs,
                           ys=ys,
                           keys= ["ID data Mahalanobis per dim", f"{self.OOD_dataname} OOD data Mahalanobis per dim"],
                           title= f"Class {class_num} 1-Dimensional Mahalanobis Distances - {self.OOD_dataname} OOD data",
                           xname= "Dimension")})

        return collated_class_dtest, collated_class_dood
    
# Makes predictions using the normal mahalanobis distance but then shows the scores taking into account the background statistics   
class Class_One_Dim_Relative_Mahalanobis(Class_One_Dim_Mahalanobis):
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
            din_class = np.mean(din_class,axis=0) # Mean of all the data points in the class
            
            dood_class = dood[i].T
            dood_class = dood_class[indices_ood ==i]
            dood_class = np.mean(dood_class,axis=0)
            
            
            collated_din_class.append(din_class)
            collated_dood_class.append(dood_class)
        

        # Change to values of 0 as this would not have any diffe
        collated_din_class = np.nan_to_num(collated_din_class,nan=0.0)
        collated_dood_class = np.nan_to_num(collated_dood_class,nan =0.0)
        return collated_din_class, collated_dood_class

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        indices_dtest, indices_dood = self.get_predictions(ftrain_norm, ftest_norm, food_norm, labelstrain)
        collated_class_dtest, collated_class_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, indices_dtest, indices_dood)
        
        #import ipdb; ipdb.set_trace()
        #xs = np.arange(len(dtest))
        #baseline = np.zeros_like(dtest)
        
        for class_num in np.unique(labelstrain):
            xs = np.arange(len(collated_class_dtest[class_num]))
            ys = [collated_class_dtest[class_num],collated_class_dood[class_num]]
            # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
            wandb.log({f"Class {class_num} 1D Relative Mahalanobis {self.OOD_dataname}" : wandb.plot.line_series(
                           xs=xs,
                           ys=ys,
                           keys= ["ID data Relative Mahalanobis per dim", f"{self.OOD_dataname} OOD data Relative Mahalanobis per dim"],
                           title= f"Class {class_num} 1-Dimensional Relative Mahalanobis Distances - {self.OOD_dataname} OOD data",
                           xname= "Dimension")})

        return collated_class_dtest, collated_class_dood
