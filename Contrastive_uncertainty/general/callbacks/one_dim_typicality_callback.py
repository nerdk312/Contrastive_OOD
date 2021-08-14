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




class One_Dim_Typicality(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

    
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use

        self.OOD_dataname = self.OOD_Datamodule.name
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)
    

    def forward_callback(self, trainer, pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        val_loader = self.Datamodule.val_dataloader()
        # Use the test transform validataion loader
        if isinstance(val_loader, tuple) or isinstance(val_loader, list):
                    _, val_loader = val_loader
                
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_val, labels_val = self.get_features(pl_module,val_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_val),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_val))
        
        auroc = get_roc_sklearn(dtest, dood)
        test_accuracy = self.mahalanobis_classification(dtest,labels_test)

        wandb.run.summary[f'1D Typicality AUROC OOD {self.OOD_dataname}'] = auroc
        wandb.run.summary[f'1D Typicality Classification'] = test_accuracy


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

    
    def get_scores(self, ftrain,fval, ftest, food, ypred,y_val_pred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        # class wise validataion data
        xc_val = [fval[y_val_pred ==i] for i in np.unique(y_val_pred)]


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

        '''Get entropy based on training data (or could get entropy using the validation data)
        # This gets the class scores for the case where there are different values present
        dtrain_class = [np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))]

        # Calculate the average 1 dimensional entropy for each different classes
        one_dim_class_entropy = [-np.mean(0.5*(dtrain_class[class_num]**2),axis= 1,keepdims=True) for class_num in range(len(cov))]
        '''

        # This gets the class scores for the case where there are different values present
        dval_class = [np.matmul(eigvectors[class_num].T,(xc_val[class_num] - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))]

        # Calculate the average 1 dimensional entropy for each different classes
        one_dim_class_entropy = [-np.mean(0.5*(dval_class[class_num]**2),axis= 1,keepdims=True) for class_num in range(len(cov))]
        
        # Calculate the average 1 dimensional mahalanobis scores for each different class
        #dtrain_class = [np.mean(dtrain_class[class_num],axis= 1) for class_num in range(len(cov))]
        
        # Inference
        # Calculate the scores for the in distribution data
        din = [np.matmul(eigvectors[class_num].T,(ftest - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))] # shape 
        # Change from values based on likelihood to entropy (no calculation of a mean as this looks at each individual data point)
        one_dim_class_nll_din = [-0.5*(din[class_num]**2) for class_num in range(len(cov))]
        # calculate  absolute deviation of scores from entropy (as well as the sum for each dimension)
        total_absolute_distance_din  = [np.sum(np.abs(one_dim_class_nll_din[class_num] - one_dim_class_entropy[class_num]),axis=0) for class_num in range(len(cov))]
        
        indices_din = np.argmin(total_absolute_distance_din,axis=0)
         # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        total_absolute_distance_din = np.min(total_absolute_distance_din,axis=0) # shape (num_data_points)
        

        # Calculate the scores for the in distribution data
        dood = [np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))] # shape 
        # Change from values based on likelihood to entropy (no calculation of a mean as this looks at each individual data point)
        one_dim_class_nll_dood = [-0.5*(dood[class_num]**2) for class_num in range(len(cov))]
        # calculate  absolute deviation of scores from entropy (as well as the sum for each dimension)
        total_absolute_distance_dood  = [np.sum(np.abs(one_dim_class_nll_dood[class_num] - one_dim_class_entropy[class_num]),axis=0) for class_num in range(len(cov))]
        
        indices_dood = np.argmin(total_absolute_distance_dood,axis=0)

        # Find min along the class dimension, so this finds the lowest 1 dimensional mahalanobis distance among the different classes for the different data points
        total_absolute_distance_dood = np.min(total_absolute_distance_dood,axis=0) # shape (num_data_points)
        
        return total_absolute_distance_din, total_absolute_distance_dood, indices_din, indices_dood

    
    # Normalises the data
    def normalise(self,ftrain,fval, ftest,food):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        fval /= np.linalg.norm(fval, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        fval = (fval - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        
        return ftrain, fval, ftest, food

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        return mahalanobis_test_accuracy
    
    def get_eval_results(self, ftrain,fval, ftest, food, labelstrain,labelsval):
        ftrain_norm, fval_norm, ftest_norm, food_norm = self.normalise(ftrain,fval, ftest, food)
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm,fval_norm,ftest_norm,food_norm,labelstrain,labelsval)
        return dtest, dood, indices_dtest, indices_dood





# Version of typicality based approach which does not use the entropy for the calculation
class One_Dim_Typicality_Class(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

    
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use

        self.OOD_dataname = self.OOD_Datamodule.name
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)
    

    def forward_callback(self, trainer, pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        val_loader = self.Datamodule.val_dataloader()
        # Use the test transform validataion loader
        if isinstance(val_loader, tuple) or isinstance(val_loader, list):
                    _, val_loader = val_loader
                
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_test))



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

    
    def get_scores(self, ftrain, ftest, food, ypred,ypredtest):
        # Get information related to the train info
        means, cov, eigvalues, eigvectors, dtrain_class = self.get_1d_train(ftrain,ypred)

        # Get the test data for a particular class of the data
        xctest = [ftest[ypredtest == i] for i in np.unique(ypredtest)]
        # Inference
        # Calculate the scores for the in distribution data


        din = [np.matmul(eigvectors[class_num].T,(xctest[class_num] - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))] # shape
        din = [np.mean(din[class_num],axis= 1) for class_num in range(len(cov))]
        # Find the deviation for din and dtrainclass
        din_deviation = [np.abs(din[i] - dtrain_class[i]) for i in range(len(cov))]
        
        dood = [np.matmul(eigvectors[class_num].T,(food - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))] # shape 
        dood = [np.mean(dood[class_num],axis= 1) for class_num in range(len(cov))]
        dood_deviation = [np.abs(dood[i] - dtrain_class[i]) for i in range(len(cov))]
        
        # Aim to have positive values which means that dood_deviation is larger than din_deviation, which means that din was close to the means of the data
        din_dood_deviation = [dood_deviation[i] - din_deviation[i] for i in range(len(cov))]
        df = pd.DataFrame(dood_deviation).T # Transpose to get the appropriate rows and columns
        # update the columns of the data
        columns = [f'Class {i}' for i in range(len(cov))]  
        df.columns = columns
        #table = wandb.Table(data=df)
        #import ipdb; ipdb.set_trace()
        num_classes = 10
        xs = list(range(len(df.index)))
        # Only access first 10 for the purpose of clutter
        ys = [df[f'Class {class_num}'].tolist() for class_num in range(num_classes)]
        
        wandb.log({f'Typicality Class Deviation {self.OOD_dataname}': wandb.plot.line_series(
            xs = xs,
            ys = ys,
            keys =columns[0:num_classes],
            title=f'Typicality Class Deviation {self.OOD_dataname}')})
        #wandb.log({f'Typicality Class Deviation {self.OOD_dataname}':table})

        return din_dood_deviation


    def get_1d_train(self, ftrain, ypred):
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

        #Get entropy based on training data (or could get entropy using the validation data)
        # This gets the class scores for the case where there are different values present
        dtrain_class = [np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(cov))]
        dtrain_class = [np.mean(dtrain_class[class_num],axis= 1) for class_num in range(len(cov))]

        return means, cov, eigvalues, eigvectors, dtrain_class
    
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

   
    
    def get_eval_results(self, ftrain, ftest, food, labelstrain,labelstest):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din_deviation = self.get_scores(ftrain_norm,ftest_norm,food_norm,labelstrain,labelstest)
        return din_deviation
