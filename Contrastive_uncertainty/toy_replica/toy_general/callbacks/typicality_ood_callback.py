import ipdb
from numpy.lib.function_base import quantile
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

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.toy_replica.toy_general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.toy_replica.toy_general.utils.pl_metrics import precision_at_k, mean

class Typicality(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

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
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
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
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_val),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_val),
            np.copy(labels_test))

        
        return fpr95,auroc,aupr

    def get_features(self, pl_module, dataloader):
        features, labels = [], []
        
        total = 0
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

    
    def get_offline_thresholds(self, ftrain, fval, ypred, yval,bootstrap_num,batch_size):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Calculate the means of the data
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        cov = [np.cov(x.T, bias=True) for x in xc]
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function formula for entropy of a gaussian distribution where |sigma| represents determinant of sigma
        #entropy = [0.5*np.log(np.linalg.det(sigma)) for sigma in cov]
        entropy = []

        for class_num in range(len(np.unique(ypred))):
            xc_class = xc[class_num]
            dtrain = np.sum(
                (xc_class - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (xc_class - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
            # Calculates approximation for the entropy
            class_entropy = -np.mean(0.5*(dtrain**2))
            entropy.append(class_entropy)
            
        # Separate into different classes
        
        thresholds = [] # List of all threshold values
        
        xval_c = [fval[yval == i] for i in np.unique(yval)]
        # Iterate through the classes
        for class_num in range(len(np.unique(yval))):
            # get the number of indices for the class
            xval_class = xval_c[class_num] 
            indices = np.arange(len(xval_c[class_num]))
            class_thresholds = [] #  List of class threshold values
            for k in range(bootstrap_num):
                vals = np.random.choice(indices, size=batch_size, replace=True)
                bootstrap_data = xval_class[vals]
                #import ipdb; ipdb.set_trace()
                # Calculates the mahalanobis distance



                dval = np.sum(
                (bootstrap_data - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (bootstrap_data - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
                
                # calculate negative log likelihood
                nll = - np.mean(0.5*(dval**2))
                threshold_k = np.abs(nll- entropy[class_num])
                #threshold_k = np.abs(np.mean(dval)- entropy[class_num])
                #threshold_k = np.abs(loglikelihood- entropy[class_num])
                class_thresholds.append(threshold_k)
            
            thresholds.append(class_thresholds)
        
        # Calculating the CDF
        final_threshold = [np.quantile(np.array(class_threshold_values),0.99) for class_threshold_values in thresholds]
        return means, cov, entropy, final_threshold


    def get_online_test_thresholds(self, means, cov, entropy, ftest, ytest, batch_size):
        test_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features
        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]
        # Iterate through the classes
        for class_num in range(len(np.unique(ytest))):
            # get the number of indices for the class
            xtest_class = xtest_c[class_num] 
            #indices = np.arange(len(xtest_c[class_num]))
            class_test_thresholds = [] #  List of class threshold values
            # obtain the num batches
            num_batches = len(xtest_class)//batch_size 
            for i in range(num_batches-1):
                ftestbatch = xtest_class[(i*batch_size):((i+1)*batch_size)]
                dtest = np.sum(
                (ftestbatch - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (ftestbatch - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)

                nll = - np.mean(0.5*(dtest**2))
                #threshold_k = np.abs(np.mean(dtest)- entropy[class_num])
                threshold_k = np.abs(nll- entropy[class_num])
                class_test_thresholds.append(threshold_k)

            test_thresholds.append(class_test_thresholds)
        



        # Treating other classes as OOD dataset for a particular class
        test_ood_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features
        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]


        # Iterate through the classes
        for class_num in range(len(np.unique(ytest))):
            # Remove a subarray related to a particular class
            import ipdb; ipdb.set_trace()
            xtest_ood = np.delete(ftest,xtest_c[class_num])
            # get the number of indices for the class
            #indices = np.arange(len(xtest_c[class_num]))
            class_test_ood_thresholds = [] #  List of class threshold values
            # obtain the num batches
            num_batches = len(xtest_ood)//batch_size 
            for i in range(num_batches-1):
                ftest_ood_batch = xtest_class[(i*batch_size):((i+1)*batch_size)]
                dtest_ood = np.sum(
                (ftest_ood_batch - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (ftest_ood_batch - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)

                nll = - np.mean(0.5*(dtest_ood**2))
                threshold_k = np.abs(nll- entropy[class_num])
                class_test_ood_thresholds.append(threshold_k)

            test_ood_thresholds.append(class_test_ood_thresholds)



        return test_thresholds, test_ood_thresholds

            
        
    

        
    def get_eval_results(self, ftrain,fval, ftest, food, labelstrain, labelsval,labels_test):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
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
        # Nawid - obtain the scores for the test data and the OOD data
        
        class_means, class_cov,class_entropy, class_quantile_thresholds = self.get_offline_thresholds(ftrain, fval,labelstrain,labelsval,50,25)
        #def get_online_test_thresholds(self, means, cov, entropy, ftest, ytest, batch_size):
        #import ipdb; ipdb.set_trace()
        self.get_online_test_thresholds(class_means,class_cov,class_entropy,ftest,labels_test, 25)

    
    