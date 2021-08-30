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



# Performs 1 dimensional typicality using a single data point
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





# Version of typicality based approach which does not use the entropy for the calculation (ORACLE VERSION)
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


# Used to calculate the 1 dimensional mahalanobis distances to see the difference in performance
class One_Dim_Typicality_Marginal_Oracle(One_Dim_Typicality_Class):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):
        super().__init__(Datamodule,OOD_Datamodule,quick_callback)
    

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
            np.copy(features_ood))



    def get_scores(self, ftrain, ftest, food):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain = self.get_1d_train(ftrain)
        # Get the test data for a particular class of the data
        
        # Inference
        # Calculate the scores for the in distribution data

        din = np.matmul(eigvectors.T,(ftest - mean).T)**2/eigvalues  # shape
        din = np.mean(din,axis= 1)

        # Find the deviation for din and dtrain
        din_deviation = np.abs(din - dtrain)


        dood = np.matmul(eigvectors.T,(food - mean).T)**2/eigvalues  # shape
        dood = np.mean(dood,axis= 1)

        # Find the deviation for din and dtrain
        dood_deviation = np.abs(dood - dtrain)

        
        # Aim to have positive values which means that dood_deviation is larger than din_deviation, which means that din was close to the means of the data
        din_dood_deviation = dood_deviation - din_deviation
        df = pd.DataFrame(dood_deviation) # Transpose to get the appropriate rows and columns
        # update the columns of the data
        
        columns = ['1D Deviation']  
        df.columns = columns
        #table = wandb.Table(data=df)
        xs = list(range(len(df.index)))
        # Only access first 10 for the purpose of clutter
        ys = [df['1D Deviation'].tolist()] 

        wandb.log({f'Typicality Oracle Marginal Deviation {self.OOD_dataname}': wandb.plot.line_series(
            xs = xs,
            ys = ys,
            keys =columns,
            title=f'Typicality Oracle Marginal Deviation {self.OOD_dataname}')})
        #wandb.log({f'Typicality Class Deviation {self.OOD_dataname}':table})

        return din_dood_deviation

    
    def get_1d_train(self, ftrain):
        # Nawid - get all the features which belong to each of the different classes
        cov = np.cov(ftrain.T, bias=True) # Cov and means part should be fine
        mean = np.mean(ftrain,axis=0,keepdims=True) # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues, eigvectors = np.linalg.eigh(cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/eigvalues
        dtrain = np.mean(dtrain,axis= 1)
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point

        #Get entropy based on training data (or could get entropy using the validation data)
        return mean, cov, eigvalues, eigvectors, dtrain
    
    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din_deviation = self.get_scores(ftrain_norm,ftest_norm,food_norm)
        return din_deviation

# Used to calculate the one dim typicality for a specific batch size
class One_Dim_Typicality_Marginal(One_Dim_Typicality_Marginal_Oracle):
    def __init__(self,Datamodule, OOD_Datamodule,quick_callback:bool=True,typicality_bsz:int=25):
        super().__init__(Datamodule,OOD_Datamodule, quick_callback)

        self.typicality_bsz= typicality_bsz
        self.summary_key = f'Unnormalized One Dim Marginal Typicality Batch Size - {self.typicality_bsz} OOD - {self.OOD_Datamodule.name}'
    
    
    def get_thresholds(self, fdata, mean, eigvalues, eigvectors,dtrain,bsz):
        thresholds = [] # List of threshold values
        num_batches = len(fdata)//bsz

        for i in range(num_batches):
            fdata_batch = fdata[(i*bsz):((i+1)*bsz)]
            ddata = np.matmul(eigvectors.T,(fdata_batch - mean).T)**2/eigvalues  # shape (dim, batch size)
            # shape (dim) average of all data in batch size
            ddata = np.mean(ddata,axis= 1)

            # Sum of the deviations of each individual dimension
            ddata_deviation = np.sum(np.abs(ddata - dtrain))
            thresholds.append(ddata_deviation)
        
        return thresholds

    def get_scores(self,ftrain, ftest, food):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain = self.get_1d_train(ftrain)

        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        din = self.get_thresholds(ftest, mean, eigvalues,eigvectors, dtrain, self.typicality_bsz)
        dood = self.get_thresholds(food, mean, eigvalues,eigvectors, dtrain, self.typicality_bsz)

        return din, dood
    
    def get_eval_results(self, ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din, dood = self.get_scores(ftrain_norm,ftest_norm,food_norm)
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC
        
        return din, dood


class One_Dim_Typicality_Normalised_Marginal(One_Dim_Typicality_Marginal):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool, typicality_bsz: int):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback=quick_callback, typicality_bsz=typicality_bsz)

        # Used to save the summary value
        self.summary_key = f'Normalized One Dim Marginal Typicality Batch Size - {self.typicality_bsz} OOD - {self.OOD_Datamodule.name}'

    # calculate the std of the 1d likelihood scores as well
    def get_1d_train(self, ftrain):
        # Nawid - get all the features which belong to each of the different classes
        cov = np.cov(ftrain.T, bias=True) # Cov and means part should be fine
        mean = np.mean(ftrain,axis=0,keepdims=True) # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues, eigvectors = np.linalg.eigh(cov)
        eigvalues = np.expand_dims(eigvalues,axis=1)
        
        dtrain = np.matmul(eigvectors.T,(ftrain - mean).T)**2/eigvalues
        # calculate the mean and the standard deviations of the different values
        dtrain_1d_mean = np.mean(dtrain,axis= 1,keepdims=True) # shape (dim,1)
        dtrain_1d_std = np.std(dtrain,axis=1,keepdims=True) # shape (dim,1)
        
        #normalised_dtrain = (dtrain - dtrain_1d_mean)
        
        # Value for a particular class
        # Vector of datapoints(embdim,num_eigenvectors) (each column is eigenvector so the different columns is the number of eigenvectors)
        # data - means is shape (B, emb_dim), therefore the matrix multiplication needs to be (num_eigenvectors, embdim), (embdim,batch) to give (num eigenvectors, Batch) and then this is divided by (num eigenvectors,1) 
        # to give (num eigen vectors, batch) different values for 1 dimensional mahalanobis distances 
        
        # I believe the first din is the list of size class, where each entry is a vector of size (emb dim, batch) where each entry of the embed dim is the 1 dimensional mahalanobis distance along that dimension, so a vector of (embdim,1) represents the mahalanobis distance of each of the n dimensions for that particular data point

        #Get entropy based on training data (or could get entropy using the validation data)
        return mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std

    def get_thresholds(self, fdata, mean, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std, bsz):
        thresholds = [] # List of threshold values
        num_batches = len(fdata)//bsz

        for i in range(num_batches):
            fdata_batch = fdata[(i*bsz):((i+1)*bsz)]
            ddata = np.matmul(eigvectors.T,(fdata_batch - mean).T)**2/eigvalues  # shape (dim, batch size)
            # Normalise the data
            ddata = (ddata - dtrain_1d_mean)/(dtrain_1d_std +1e-10) # shape (dim, batch)

            # shape (dim) average of all data in batch size
            ddata = np.mean(ddata,axis= 1) # shape : (dim)
            
            # Sum of the deviations of each individual dimension
            ddata_deviation = np.sum(np.abs(ddata))

            thresholds.append(ddata_deviation)
        
        return thresholds

    
    def get_scores(self,ftrain, ftest, food):
        # Get information related to the train info
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain)

        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        din = self.get_thresholds(ftest, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std, self.typicality_bsz)
        dood = self.get_thresholds(food, mean, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std, self.typicality_bsz)

        return din, dood


# Perform typicality using a point value of a single class datapoint
class Point_One_Dim_Class_Typicality_Normalised(pl.Callback):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use

        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key =  f'Normalized Point One Dim Class Typicality OOD - {self.OOD_Datamodule.name}'

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
    
    def get_1d_train(self,ftrain, ypred):
        
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        covs = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        eigvalues = []
        eigvectors = []
        
        dtrain_1d_mean = [] # 1D mean for each class
        dtrain_1d_std = [] # 1D std for each class
        for class_num, class_cov in enumerate(covs):
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)

            # Get the distribution of the 1d Scores from the certain class, which involves seeing the one dimensional scores for a specific class and calculating the mean and the standard deviation
            dtrain_class = np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/eigvalues[class_num]
            dtrain_1d_mean.append(np.mean(dtrain_class, axis= 1, keepdims=True))
            dtrain_1d_std.append(np.std(dtrain_class, axis= 1, keepdims=True))        
        
        return means, covs, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std
    
    def get_thresholds(self, fdata, means, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std):
        ddata = [np.matmul(eigvectors[class_num].T,(fdata - means[class_num]).T)**2/eigvalues[class_num] for class_num in range(len(means))] # Calculate the 1D scores for all the different classes 
        
        # obtain the normalised the scores for the different classes
        ddata = [ddata[class_num] - dtrain_1d_mean[class_num]/(dtrain_1d_std[class_num] + +1e-10) for class_num in range(len(means))] # shape (dim, batch)

        # Obtain the sum of absolute normalised scores
        scores = [np.sum(np.abs(ddata[class_num]),axis=0) for class_num in range(len(means))]
        # Obtain the scores corresponding to the lowest class
        ddata = np.min(scores,axis=0)

        return ddata

    def get_scores(self,ftrain,ftest, food, labelstrain):
        means, covs, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std = self.get_1d_train(ftrain, labelstrain)

        # Inference
        # Calculate the scores for the in-distribution data and the OOD data
        din = self.get_thresholds(ftest, means, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        dood = self.get_thresholds(food, means, eigvalues,eigvectors, dtrain_1d_mean,dtrain_1d_std)
        return din, dood
    
    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        din, dood = self.get_scores(ftrain, ftest, food, labelstrain)
        AUROC = get_roc_sklearn(din, dood)
        wandb.run.summary[self.summary_key] = AUROC



# Performs analysis of the relative mahalanaobis distance scores
class Point_One_Dim_Relative_Class_Typicality_Analysis(Point_One_Dim_Class_Typicality_Normalised):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)


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
        

        # Used to get the predcitions for te OOD data points - involves normalising the data
        din, dood, indices_din, indices_dood = self.get_predictions(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        
        # Does not involve normalising the data
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_test),
            np.copy(indices_dood))

    
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
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood

    # normalise scores to get the predictions of food
    def normalise(self,ftrain,ftest,food):
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
        
        return ftrain, ftest,food

    def get_predictions(self,ftrain, ftest,food,labelstrain):
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        return dtest, dood, indices_dtest, indices_dood

    def get_1d_train(self,ftrain, ypred): 
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        covs = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        ####### Background information ##########
        background_cov = np.cov(ftrain.T, bias=True)
        background_mean = np.mean(ftrain,axis=0,keepdims=True)
        background_eigvalues, background_eigvectors = np.linalg.eigh(background_cov)
        background_eigvalues =  np.expand_dims(background_eigvalues,axis=1)

        eigvalues = []
        eigvectors = []
            
        dtrain_1d_mean = [] # 1D mean for each class
        dtrain_1d_std = [] # 1D std for each class
        all_dtrain_class = [] # 1d scores for each class

        background_class_1d_mean = [] # 1D mean for each class
        background_class_1d_std = [] # 1D std for each class
        all_background_train_class = [] # scores for the background
        for class_num, class_cov in enumerate(covs):
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of
            
            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals,axis=1))
            eigvectors.append(class_eigvectors)

            # Get the distribution of the 1d Scores from the certain class, which involves seeing the one dimensional scores for a specific class and calculating the mean and the standard deviation
            dtrain_class = np.matmul(eigvectors[class_num].T,(xc[class_num] - means[class_num]).T)**2/eigvalues[class_num]
            dtrain_1d_mean.append(np.mean(dtrain_class, axis= 1, keepdims=True))
            dtrain_1d_std.append(np.std(dtrain_class, axis= 1, keepdims=True))
            all_dtrain_class.append(dtrain_class)

            # Information related to the background statistics of each class
            background_class = np.matmul(background_eigvectors.T,(xc[class_num] - background_mean).T)**2/background_eigvalues
            background_class_1d_mean.append(np.mean(background_class, axis= 1, keepdims=True))
            background_class_1d_std.append(np.std(background_class, axis= 1, keepdims=True))
            all_background_train_class.append(background_class)

            class_semantic_information = means, covs, eigvalues, eigvectors,all_dtrain_class, dtrain_1d_mean, dtrain_1d_std
            class_background_information = background_mean, background_cov,  background_eigvalues, background_eigvectors, all_background_train_class, background_class_1d_mean, background_class_1d_std
        
        return class_semantic_information, class_background_information
    # Continue by getting the test and OOD statistics of the data
    
    def get_statistics(self, fdata,labels_data,train_labels, class_semantic_information,class_background_information):
        xc = [fdata[labels_data == i] for i in np.unique(train_labels)] # Need to use the train labels to get all the labels
        
        semantic_means, semantic_cov, semantic_eigvalues, semantic_eigvectors,all_semantic_train_class, semantic_dtrain_1d_mean, semantic_dtrain_1d_std = class_semantic_information     
        background_mean, background_cov,  background_eigvalues, background_eigvectors, all_background_train_class, background_class_1d_mean, background_class_1d_std =  class_background_information        

        semantic_raw_ddata = [np.matmul(semantic_eigvectors[class_num].T,(xc[class_num] - semantic_means[class_num]).T)**2/semantic_eigvalues[class_num] for class_num in range(len(xc))] # Calculate the 1D scores for all the different classes         
        # obtain the normalised the scores for the different classes (requires obtaining the absolute value) which is important for replacing nans in the data
        semantic_normalized_ddata = [np.abs(semantic_raw_ddata[class_num] - semantic_dtrain_1d_mean[class_num]/(semantic_dtrain_1d_std[class_num] +1e-10)) for class_num in range(len(xc))] # shape (dim, batch)

        background_raw_ddata = [np.matmul(background_eigvectors.T,(xc[class_num] - background_mean).T)**2/background_eigvalues for class_num in range(len(xc))] # Calculate the 1D scores for all the different classes         
        # obtain the normalised the scores for the different classes
        background_normalized_ddata = [np.abs(background_raw_ddata[class_num] - background_class_1d_mean[class_num]/(background_class_1d_std[class_num]  +1e-10)) for class_num in range(len(xc))] # shape (dim, batch)

        return semantic_raw_ddata, semantic_normalized_ddata, background_raw_ddata, background_normalized_ddata

    def get_analysis_scores(self,ftrain,ftest, food, labelstrain,labelstest,predictedood):
        class_semantic_information, class_background_information = self.get_1d_train(ftrain, labelstrain)
        
        semantic_raw_din, semantic_normalized_din, background_raw_din, background_normalized_din = self.get_statistics(ftest,labelstest,labelstrain,class_semantic_information, class_background_information)
        semantic_raw_dood, semantic_normalized_dood, background_raw_dood, background_normalized_dood = self.get_statistics(food,predictedood,labelstrain,class_semantic_information, class_background_information)        
        return semantic_raw_din, semantic_normalized_din, background_raw_din, background_normalized_din, semantic_raw_dood, semantic_normalized_dood, background_raw_dood, background_normalized_dood 

    def get_eval_results(self, ftrain, ftest, food, labelstrain,labelstest,predictedood):
        semantic_raw_din, semantic_normalized_din, background_raw_din, background_normalized_din, semantic_raw_dood, semantic_normalized_dood, background_raw_dood, background_normalized_dood  = self.get_analysis_scores(ftrain,ftest, food, labelstrain,labelstest,predictedood)

        self.datasaving(semantic_normalized_din,background_normalized_din,semantic_normalized_dood, background_normalized_dood)
    
    def datasaving(self, semantic_normalized_din,  background_normalized_din, semantic_normalized_dood,background_normalized_dood):
        num_classes = len(semantic_normalized_din)
           
        semantic_din = [pd.DataFrame(semantic_normalized_din[class_num].T) for class_num in range(num_classes)]
        background_din =  [pd.DataFrame(background_normalized_din[class_num].T) for class_num in range(num_classes)]

        semantic_dood = [pd.DataFrame(semantic_normalized_dood[class_num].T) for class_num in range(num_classes)]
        background_dood =  [pd.DataFrame(background_normalized_dood[class_num].T) for class_num in range(num_classes)]
        
        all_df = copy.deepcopy(semantic_din)
        # Need to add the lists into a single existing list
        all_df.extend(background_din)
        all_df.extend(semantic_dood)
        all_df.extend(background_dood)
        
        # Concatenate all the dataframes (which places nans in situations where the columns have different lengths)
        all_df = pd.concat(all_df,axis=1)
        all_df = all_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        
        semantic_din_columns = [f'Semantic typicality scores ID Class {i}' for i in range(num_classes)]
        background_din_columns = [f'Background typicality scores ID Class {i}' for i in range(num_classes)]
        semantic_dood_columns = [f'Semantic typicality scores OOD {self.OOD_dataname} Class {i}' for i in range(num_classes)]
        background_dood_columns = [f'Background typicality scores OOD {self.OOD_dataname} Class {i}' for i in range(num_classes)]
        
        all_df.columns = semantic_din_columns + background_din_columns + semantic_dood_columns + background_dood_columns

        analysis_table = wandb.Table(data = all_df)
        wandb.log({'Practice':analysis_table})


        # Need to get the correct number of columns for the data. The number of columns should be the number of class x number of dimensions (10 x128) x 4 (due too having 4 measurements )