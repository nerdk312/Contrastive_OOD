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
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_fpr, get_pr_sklearn, get_roc_plot, get_roc_sklearn, table_saving


class Comparison_practice(pl.Callback):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
   
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name

        self.vector_level = 'fine'
        self.label_level = 'coarse'
    def on_validation_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()


        features_train_fine, labels_train_fine = self.get_features_HM(pl_module, train_loader,'fine')
        features_fine_HM, labels_fine_HM = self.get_features_HM(pl_module, test_loader,'fine') 
        features_ood_fine, labels_ood_fine = self.get_features_HM(pl_module, ood_loader, 'fine')

        '''
        features_fine_non_HM, label_coarse_non_HM =  self.get_features_non_HM(pl_module, test_loader)
        features_coarse_HM, labels_coarse_HM = self.get_features_HM(pl_module, test_loader,'coarse')
        
        print('features non HM', features_fine_non_HM)
        print('features HM', features_fine_HM)
        print('labels non HM', label_coarse_non_HM)
        print('labels HM', labels_coarse_HM)
        print('feature diff', features_fine_non_HM - features_fine_HM)
        print('labels diff', label_coarse_non_HM - labels_coarse_HM)
        
        # Test that normalisation works
        print('train norm difference',ftrain_norm -features_train_fine)
        print('test norm difference', ftest_norm -features_fine_HM)
        
        
        # Test that the unconditional score case works
        print('nonHM_indices_din',nonHM_indices_din)
        print('HM_indices_din',HM_indices_din)

        print('din diff', nonHM_din - HM_din)
        print('dood diff', nonHM_dood - HM_dood)
        print('indices_in diff', nonHM_indices_din - HM_indices_din)
        print('indices_ood diff', nonHM_indices_dood - HM_indices_dood)
        '''

        ftrain_norm, ftest_norm, food_norm = self.normalise(features_train_fine, features_fine_HM, features_ood_fine)
        nonHM_din, nonHM_dood, nonHM_indices_din, nonHM_indices_dood =self.get_scores_non_HM(ftrain_norm, ftest_norm, food_norm,labels_train_fine)
        
        HM_din, HM_dood, HM_indices_din, HM_indices_dood = self.get_scores_HM(ftrain_norm, ftest_norm, food_norm, labels_train_fine)
        


    def get_scores_non_HM(self,ftrain, ftest, food, ypred):
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

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood
    
    def get_scores_HM(self, ftrain, ftest, food, ypred, ptest_index = None, pood_index=None): # Add additional variables for the parent index
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        # Obtain the mahalanobis distance scores for the different classes on the train data
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
        
        
        # Obtain the mahalanobis distance scores for the different classes on the test data
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
        
        din, indices_din = self.get_conditional_scores(din,ptest_index)
        dood, indices_dood = self.get_conditional_scores(dood,pood_index)
        
        return din, dood, indices_din, indices_dood
    
    def get_conditional_scores(self,ddata, prev_indices=None):
        # import ipdb; ipdb.set_trace()
        if prev_indices is not None: # index of the previous test values
            coarse_test_mapping =  self.Datamodule.coarse_mapping.numpy()
            ddata = np.stack(ddata,axis=1) # stacks the array to make a (batch,num_classes) array
            collated_ddata = []
            collated_indices = []
            # Go throuhg each datapoint hierarchically
            for i,sample_distance in enumerate(ddata):
                # coarse_test_mapping==ptest_index[i]] corresponds to a boolean mask placed on sample to get only the values of interest
                conditioned_distance = sample_distance[coarse_test_mapping==prev_indices[i]] # Get the data point which have the same superclass
                # Obtain the smallest value for the conditioned distances
                min_conditioned_distance = np.min(conditioned_distance)
                sample_index = np.where(sample_distance == min_conditioned_distance)[0][0] # Obtain the index from the datapoint to get the fine class label

                collated_ddata.append(min_conditioned_distance)
                collated_indices.append(sample_index)

            ddata = np.array(collated_ddata)
            indices_ddata = np.array(collated_indices)
        else:    
            indices_ddata = np.argmin(ddata,axis = 0)  
            ddata = np.min(ddata, axis=0) # Nawid - calculate the minimum distance 

        return ddata, indices_ddata





    def normalise(self, ftrain, ftest, food):
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


    def get_features_non_HM(self, pl_module, dataloader):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][self.label_level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
            
            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][self.vector_level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
            
        return np.array(features), np.array(labels)
    
    def get_features_HM(self, pl_module, dataloader, level):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]

            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
        #import ipdb; ipdb.set_trace()              

        return np.array(features), np.array(labels)
    
