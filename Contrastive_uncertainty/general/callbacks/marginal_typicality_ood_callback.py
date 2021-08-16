#import ipdb
from numpy.lib.function_base import average, quantile
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





# Performs the typicality test between the ID test data and the OOD data
class Marginal_Typicality_OOD_detection(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True,
        typicality_bsz:int = 25):
        
        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        #import ipdb; ipdb.set_trace()
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.vector_level = vector_level
        self.label_level = label_level
        
        self.OOD_dataname = self.OOD_Datamodule.name

        self.typicality_bsz = typicality_bsz

        
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
            np.copy(features_ood))
            

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

    # Get the statistics from the training information which is present
    def get_statistics(self, ftrain):
        mean = np.mean(ftrain, axis=0,keepdims=True)
        cov = np.cov(ftrain.T, bias=True)
        
        dtrain = np.sum(
                (ftrain - mean) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftrain - mean).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
        # Calculates approximation for the entropy
        entropy = -np.mean(0.5*(dtrain**2))
        return mean, cov, entropy

    # Used to calculate the thresholds for the general case
    '''
    def get_thresholds(self, fdata, mean, cov, entropy):
        thresholds = [] #  List of class threshold values
        # obtain the num batches
        num_batches = len(fdata)//self.typicality_bsz
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.typicality_bsz):((i+1)*self.typicality_bsz)]
            ddata = np.sum(
            (fdata_batch - mean) # Nawid - distance between the data point and the mean
            * (
                np.linalg.pinv(cov).dot(
                    (fdata_batch - mean).T
                ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
            ).T,
            axis=-1)

            nll = - np.mean(0.5*(ddata**2))
            threshold_k = np.abs(nll- entropy)
            thresholds.append(threshold_k)
        
        return thresholds
    '''
    def get_thresholds(self, fdata, mean, cov, entropy, bsz):
        thresholds = [] #  List of class threshold values
        # obtain the num batches
        num_batches = len(fdata)//bsz
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*bsz):((i+1)*bsz)]
            ddata = np.sum(
            (fdata_batch - mean) # Nawid - distance between the data point and the mean
            * (
                np.linalg.pinv(cov).dot(
                    (fdata_batch - mean).T
                ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
            ).T,
            axis=-1)

            nll = - np.mean(0.5*(ddata**2))
            threshold_k = np.abs(nll- entropy)
            thresholds.append(threshold_k)
        
        return thresholds
       
    def get_eval_results(self, ftrain, ftest, food):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        mean, cov, entropy = self.get_statistics(ftrain_norm)
        '''
        test_thresholds = self.get_thresholds(ftest_norm,mean, cov, entropy)
        ood_thresholds = self.get_thresholds(food_norm, mean, cov, entropy)
        AUROC =get_roc_sklearn(test_thresholds, ood_thresholds)
        '''
        bszs = [10,15,20,25]
        self.data_saving(mean,cov,entropy,ftest_norm,food_norm, bszs,f'Marginal Typicality OOD {self.OOD_dataname} Table')

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
    
    def data_saving(self,mean, cov, entropy, ftest, food, bszs, table_name):
        table_data = {'Batch Size': [],'AUROC': []}
        
        for bsz in bszs:
            test_thresholds = self.get_thresholds(ftest,mean, cov, entropy, bsz)
            ood_thresholds = self.get_thresholds(food, mean, cov, entropy, bsz)
            auroc = get_roc_sklearn(test_thresholds, ood_thresholds)

            table_data['Batch Size'].append(bsz)
            table_data['AUROC'].append(round(auroc,3))
            table_df = pd.DataFrame(table_data)
            ood_table = wandb.Table(dataframe=table_df)

            wandb.log({table_name: ood_table})
        
        

    '''
        # Class conditional thresholds for the correct class
        bszs = [1,2,3,5,10]
        self.OVR_AUROC_saving(class_means,class_cov, class_entropy,
        ftest_norm,food_norm,labels_test,bszs,'Batch Size',f'Typicality One Vs OOD Rest Average Batch Sizes {self.OOD_dataname}')
        
    def OVR_AUROC_saving(self,class_means, class_cov,class_entropy,ftest, food,labels,bszs,table_name,wandb_name):
        table_data = {table_name: [],'Average AUROC': []}
        for bsz in bszs:
            test_thresholds = self.get_online_test_thresholds(class_means,class_cov,class_entropy,ftest,labels,bsz)
            # Class conditional thresholds using OOD test data
            ood_thresholds = self.get_online_ood_thresholds(class_means, class_cov,class_entropy, food,bsz)

            class_aurocs = []
            for class_num in range(len(test_thresholds)):
                class_auroc = get_roc_sklearn(test_thresholds[class_num], ood_thresholds[class_num])
                class_aurocs.append(class_auroc)
            
            table_data['Average AUROC'].append(round(statistics.mean(class_aurocs),2))
            table_data[table_name].append(bsz)
        
        table_df = pd.DataFrame(table_data)
        ood_table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name: ood_table})
    '''


# Performs the typicality test between the ID test data and the OOD data using both the entropy as well as the average likelihood
class Marginal_Typicality_entropy_mean(pl.Callback):
    def __init__(self, Datamodule, OOD_Datamodule, quick_callback: bool, typicality_bsz: int):
        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        #import ipdb; ipdb.set_trace()
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use        
        self.OOD_dataname = self.OOD_Datamodule.name

        self.typicality_bsz = typicality_bsz
        # Calculates both the entropy and the average likelihood for the approach
        self.summary_key = f'Marginal Typicality Entropy Average Likelihood Batch Size - {self.typicality_bsz} OOD - {self.OOD_Datamodule.name}'

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    
    # Performs all the computation in the callback
    def forward_callback(self, trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        
        # Use the test transform validataion loader
        
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood))
            
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

    # Get the statistics from the training information which is present
    def get_statistics(self, ftrain):
        mean = np.mean(ftrain, axis=0,keepdims=True)
        cov = np.cov(ftrain.T, bias=True)
        
        dtrain = np.sum(
                (ftrain - mean) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftrain - mean).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
        # Calculates approximation for the entropy
        entropy = -np.mean(0.5*(dtrain**2))
        average_likelihood = np.mean(dtrain)
        #average_likelihood = np.mean(dtrain, keepdims=True)
        return mean, cov, entropy, average_likelihood

    # Used to calculate the thresholds for the general case
    def get_thresholds(self, fdata, mean, cov, entropy,average_likelihood, bsz):
        thresholds_entropy = [] #  List of class threshold values
        thresholds_mean = []
        # obtain the num batches
        num_batches = len(fdata)//bsz
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*bsz):((i+1)*bsz)]
            ddata = np.sum(
            (fdata_batch - mean) # Nawid - distance between the data point and the mean
            * (
                np.linalg.pinv(cov).dot(
                    (fdata_batch - mean).T
                ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
            ).T,
            axis=-1)

            nll = - np.mean(0.5*(ddata**2))
            threshold_entropy_k = np.abs(nll- entropy)
            thresholds_entropy.append(threshold_entropy_k)

            # Threshold using average likelihood approach 
            average_data_ll = np.mean(ddata)
            thresholds_mean_k = np.abs(average_data_ll - average_likelihood)
            thresholds_mean.append(thresholds_mean_k)
            
        return thresholds_entropy, thresholds_mean
       
    def get_eval_results(self, ftrain, ftest, food):
        """
            None.
        """
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        mean, cov, entropy, average_likelihood = self.get_statistics(ftrain_norm)
        
        test_thresholds_entropy, test_thresholds_average_ll = self.get_thresholds(ftest_norm,mean, cov, entropy,average_likelihood, self.typicality_bsz)
        ood_thresholds_entropy, ood_thresholds_average_ll = self.get_thresholds(food_norm, mean, cov, entropy,average_likelihood, self.typicality_bsz)
        
        # AUROC using entropy approach and AUROC using mean likelihood approach (not the mean of the AUROC)
        AUROC_entropy = round(get_roc_sklearn(test_thresholds_entropy, ood_thresholds_entropy),3)
        AUROC_average_ll = round(get_roc_sklearn(test_thresholds_average_ll, ood_thresholds_average_ll),3)
        AUROC = [AUROC_entropy, AUROC_average_ll]
        wandb.run.summary[self.summary_key] = AUROC

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