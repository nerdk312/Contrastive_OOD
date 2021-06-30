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

        dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
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
        #import ipdb; ipdb.set_trace()              

        return np.array(features), np.array(labels)
    
    def get_scores(self, ftrain, ftest, food, ypred):
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
        
         

        cov = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)
        eigvalues = []
        eigvectors = []
        for class_cov in cov:
            class_eigvals, class_eigvectors =  np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        output = [np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))]     
        output_values = np.min(output,axis=0) # Find min along the class dimension
        mean_dims = np.mean(output_values, axis=1) # Find the mean of all the data points

        array = []
        
        array1 = np.array([[3,4],[1,2],[5,6],[0,9]])
        array2 = np.array([[1,2],[3,4],[5,6],[7,8]])
        array3 = np.array([[3,5],[1,3],[4,6],[1,7]])
        array.append(array1)
        array.append(array2)
        array.append(array3)
        import ipdb; ipdb.set_trace()



        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        return dtest, dood, indices_dtest, indices_dood
