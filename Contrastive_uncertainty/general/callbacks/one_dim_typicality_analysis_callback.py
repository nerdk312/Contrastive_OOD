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


    def get_scores(self, ftrain_norm, ftest_norm, food_norm):
        return super().get_scores(ftrain_norm, ftest_norm, food_norm)

    def datasaving(self, din_std_1d):
        num_dimensions = len(din_std_1d)
        flattened_1d_std_values = din_std_1d.flatten()
        dimensions = np.arange(0,num_dimensions)
        repeated_dims = np.tile(dimensions,len(flattened_1d_std_values)//num_dimensions) 
        
        # Add the data and the dimensions together
        collated_data = np.stack((flattened_1d_std_values,repeated_dims),axis=1)
        columns = ['ID Standard Deviation Values','Dimension']
        

        df = pd.DataFrame(collated_data,columns=columns) #  Need to transpose the column to get it into the correct shape
        table = wandb.Table(dataframe=df)
        wandb.log({'ID Data Augmentation 1D Standard Deviation Values':table})

    def get_eval_results(self,ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        din_std_1d, dood_std_1d = self.get_scores(ftrain_norm,ftest_norm,food_norm)
        self.datasaving(din_std_1d)
            
# Used to check the order of dimensions and the magnitude of the eigenvalues
class eigenvalue_order_check(Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)

    
    def forward_callback(self, trainer, pl_module):
        return super().forward_callback(trainer, pl_module)

    def normalise(self, ftrain, ftest, food):
        return super().normalise(ftrain, ftest, food)
    
    # Used to obtain the eigenvalues
    def get_1d_train(self, ftrain):
        return super().get_1d_train(ftrain)
    
    # Normalise the data
    def get_eval_results(self,ftrain, ftest, food):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        mean, cov, eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std= self.get_1d_train(ftrain_norm)        

        # Saves the line plot of the data
        self.datasaving(eigvalues)
        # Need to concatenate the different tensors to make the dataframe

    def datasaving(self, eigvalues):
        num_dimensions = len(eigvalues)
        columns = ['Eigenvalues']
        df = pd.DataFrame(eigvalues, columns=columns)
        xs = list(range(num_dimensions))
        ys = [df['Eigenvalues'].to_list()]

        wandb.log({f'Eigenvalue order check': wandb.plot.line_series(
            xs = xs,
            ys = ys,
            keys =columns,
            title=f'Eigenvalue order check')})


# Calculates the KL divergence of the background and the individual class
class One_Dim_Background_Class_divergence_analysis(Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool):
        super().__init__(Datamodule, OOD_Datamodule, quick_callback)


    def forward_callback(self, trainer, pl_module):

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)

        self.get_eval_results(
            np.copy(features_train),
            np.copy(labels_train))


    def get_features(self, pl_module, dataloader):
        return super().get_features(pl_module, dataloader)
    
    # normalise scores to get the predictions of food
    def normalise(self,ftrain):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        
        return ftrain
    
    def get_1d_train(self, ftrain, ypred):
        ####### Background information ##########
        background_cov = np.cov(ftrain.T, bias=True)
        background_mean = np.mean(ftrain,axis=0,keepdims=True)
        background_eigvalues, background_eigvectors = np.linalg.eigh(background_cov)
        background_eigvalues =  np.expand_dims(background_eigvalues,axis=1)

        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        covs = [np.cov(x.T, bias=True) for x in xc] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        eigvalues = []
        eigvectors = []

        dtrain_1d_mean = [] # 1D mean for each class
        dtrain_1d_std = [] # 1D std for each class
        all_dtrain_class = [] # 1d scores for each class

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
        background_class = np.matmul(background_eigvectors.T,(ftrain - background_mean).T)**2/background_eigvalues
        background_class_1d_mean = np.mean(background_class, axis= 1, keepdims=True)
        background_class_1d_std = np.std(background_class, axis= 1, keepdims=True)
        
        return eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std, background_class_1d_mean, background_class_1d_std

    def get_1d_kl(self,dtrain_1d_mean, dtrain_1d_std, background_class_1d_mean, background_class_1d_std):
        num_dimensions = len(background_class_1d_std)
        num_classes = len(dtrain_1d_mean)
        kl_values = []
        dimension_values = []

        for i in range(num_dimensions):
            background_1d_gaussian = torch.distributions.normal.Normal(torch.tensor([background_class_1d_mean[i]]),torch.tensor([background_class_1d_std[i]]))
            class_1d_gaussian = [torch.distributions.normal.Normal(torch.tensor([dtrain_1d_mean[class_num][i]]),torch.tensor([dtrain_1d_std[class_num][i]])) for class_num in range(num_classes)]
            background_class_1d_kl = [torch.distributions.kl.kl_divergence(background_1d_gaussian,class_1d_gaussian[class_num]).item() for class_num in range(num_classes)]
            dimension_val = [i for class_num in range(num_classes)]
            # Extend used to a list to the end of another list
            # Caclculate the mean directly for the case of CIFAR100 as I cannot place enough data for the table
            if num_classes > 30: 
                kl_values.append(np.mean(background_class_1d_kl))
                dimension_values.append(i)
            else:    
                kl_values.extend(background_class_1d_kl)
                dimension_values.extend(dimension_val)

            '''
            for class_num in range(num_classes):
                class_1d_gaussian = torch.distributions.normal.Normal(torch.tensor([dtrain_1d_mean[class_num][i]]),torch.tensor([dtrain_1d_std[class_num][i]]))

                background_class_1d_kl = torch.distributions.kl.kl_divergence(background_1d_gaussian,class_1d_gaussian)

                kl_values.append(background_class_1d_kl)
                dimension_values.append(i)
            '''
        
        return kl_values, dimension_values



    def datasaving(self, kl_values, dimension_values):
        
        collated_data = np.stack((kl_values,dimension_values),axis=1)
        # Add the data and the dimensions together
        columns = ['1D Total Class KL Values','Dimension']
        df = pd.DataFrame(collated_data,columns=columns) #  Need to transpose the column to get it into the correct shape
        table = wandb.Table(dataframe=df)
        wandb.log({'1D Total Class KL Values':table})

        
        
    
    def get_eval_results(self,ftrain,labelstrain):
        ftrain_norm= self.normalise(ftrain)
        
        eigvalues, eigvectors, dtrain_1d_mean, dtrain_1d_std, background_class_1d_mean, background_class_1d_std = self.get_1d_train(ftrain_norm, labelstrain)
        kl_values, dimension_values = self.get_1d_kl(dtrain_1d_mean, dtrain_1d_std, background_class_1d_mean, background_class_1d_std)

        self.datasaving(kl_values, dimension_values)






