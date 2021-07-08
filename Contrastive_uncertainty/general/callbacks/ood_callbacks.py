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


from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean

from Contrastive_uncertainty.general.callbacks.compare_distributions import ks_statistic, ks_statistic_kde, js_metric, kl_divergence



class Mahalanobis_OOD(pl.Callback):
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
        
        self.log_name = "Mahalanobis"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.vector_level = vector_level
        self.label_level = label_level
        #import ipdb; ipdb.set_trace()
        
        self.OOD_dataname = self.OOD_Datamodule.name
    
    def on_validation_epoch_end(self, trainer, pl_module):
        #import ipdb; ipdb.set_trace()
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self, trainer, pl_module):
        #import ipdb; ipdb.set_trace()
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        #import ipdb; ipdb.set_trace()
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        num_classes = max(labels_train+1)
        fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            num_classes)

        # Obtain score and plot the score
        mahalanobis_test_accuracy = self.mahalanobis_classification(indices_dtest, labels_test)
        wandb.log({f'Mahalanobis Classification: {self.vector_level}: {self.label_level}': mahalanobis_test_accuracy})

        # Calculates the mahalanobis distance using unsupervised approach
        # Plots the curve if during testing phase
        if trainer.testing:
            get_roc_plot(dtest,dood, self.OOD_dataname)  
        return fpr95,auroc,aupr

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        return mahalanobis_test_accuracy

    def get_features(self, pl_module, dataloader):
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
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain, num_clusters):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        self.log_confidence_scores(dtest,dood,num_clusters)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)

        wandb.log({self.log_name + f' AUROC: {self.vector_level} vector: {num_clusters} classes: {self.OOD_dataname}': auroc})
        wandb.log({self.log_name + f' AUPR: {self.vector_level} vector: {num_clusters} classes: {self.OOD_dataname}': aupr})
        wandb.log({self.log_name + f' FPR: {self.vector_level} vector: {num_clusters} classes: {self.OOD_dataname}': fpr95})
        
        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood
    
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

    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,Dtest,DOOD,num_clusters):  
        confidence_test = Dtest
        confidence_OOD  = DOOD
         # histogram of the confidence scores for the true data
        true_data = [[s] for s in confidence_test]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        # Examine if the centroid was obtained in supervised or unsupervised manner
            
        true_histogram_name = self.true_histogram + f': Supervised: {self.vector_level} vector: {num_clusters} classes'
        ood_histogram_name = self.ood_histogram + f': Supervised: {self.vector_level} vector:{num_clusters} classes: {self.OOD_dataname}'
       
        #import ipdb; ipdb.set_trace()
        wandb.log({true_histogram_name: wandb.plot.histogram(true_table, "scores",title=true_histogram_name)})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in confidence_OOD]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({ood_histogram_name: wandb.plot.histogram(ood_table, "scores",title=ood_histogram_name)})


# Calculate the Mahalanobis scores for all the dataset
class Mahalanobis_OOD_Datasets(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodules,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        # Several different datamodules
        self.OOD_Datamodules = OOD_Datamodules
        for i in range(len(OOD_Datamodules)):
            # use the setup for each datamodule
            self.OOD_Datamodules[i].test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
            self.OOD_Datamodules[i].setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        #import ipdb; ipdb.set_trace()
        
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.vector_level = vector_level
        self.label_level = label_level
        
        #self.OOD_dataname = self.OOD_Datamodule.name
    '''
    def on_validation_epoch_end(self, trainer, pl_module):
        #import ipdb; ipdb.set_trace()
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    '''
    def on_test_epoch_end(self, trainer, pl_module):
        #import ipdb; ipdb.set_trace()
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
                
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        
        
        # Iterate through all the different OOD datamodules to get the labels of the OOD dataset
        collated_features_ood = []
        #labels_ood = []

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        # Collate all the OOD features for the different OOD datamodules
        for i in range(len(self.OOD_Datamodules)):
            ood_loader = self.OOD_Datamodules[i].test_dataloader()
            features_ood, _ = self.get_features(pl_module,ood_loader)
            collated_features_ood.append(features_ood)
        #import ipdb; ipdb.set_trace()
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            collated_features_ood,
            np.copy(labels_train))

    def get_features(self, pl_module, dataloader):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        #import ipdb; ipdb.set_trace()
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
    
    
    def get_scores(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        #import ipdb; ipdb.set_trace()
        dtrain = [
            np.sum(
                (ftrain - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (ftrain - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]
        #import ipdb; ipdb.set_trace()
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
        #import ipdb; ipdb.set_trace()
        collated_dood = []
        collated_indices_dood = []
        for i in range(len(self.OOD_Datamodules)):
            dood = [
                np.sum(
                    (food[i] - np.mean(x, axis=0, keepdims=True))
                    * (
                        np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                            (food[i] - np.mean(x, axis=0, keepdims=True)).T
                        )
                    ).T,
                    axis=-1,
                )
                for x in xc # Nawid- this calculates the score for all the OOD examples 
            ]
            #import ipdb; ipdb.set_trace()
            indices_dood = np.argmin(dood, axis=0)
            dood = np.min(dood, axis=0)
            
            collated_dood.append(dood)
            collated_indices_dood.append(indices_dood)

        # Calculate the indices corresponding to the values
        indices_dtrain = np.argmin(dtrain,axis=0)
        indices_din = np.argmin(din,axis = 0)
    

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dtrain = np.min(dtrain,axis=0) # caclulates the min distance for the train dataset

        return dtrain, din, collated_dood, indices_dtrain, indices_din, collated_indices_dood
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)

        # Normalise each OOD features in the list sequentially
        dataset_names = []
        dataset_names.append(f'{self.Datamodule.name}-train')
        dataset_names.append(f'{self.Datamodule.name}-test')
        for i in range(len(self.OOD_Datamodules)):
            food[i] /= np.linalg.norm(food[i], axis=-1, keepdims=True) + 1e-10
            food[i] = (food[i] - m) / (s + 1e-10)
            dataset_names.append(self.OOD_Datamodules[i].name)

            
        # Stack the OOD scores to enable it to be changed to dataframe
        # https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

        # Nawid - obtain the scores for the test data and the OOD data
        dtrain,dtest, collated_dood, indices_dtrain, indices_dtest, collated_indices_dood = self.get_scores(ftrain, ftest, food, labelstrain)
        # Make a new list for all the different distances
        collated_data = [dtrain] +[dtest] + collated_dood
        # Make a dictionary which contains the names and the mappings which I pass into the sns function
        collated_dict = {}
        num_bins = 50
        for i in range(len(collated_data)):
            collated_dict.update({dataset_names[i]:collated_data[i]})
        data_label = 'Datasets'
        
        # Plots the counts, probabilities as well as the kde
        count_histogram(collated_dict,num_bins,data_label)
        probability_histogram(collated_dict,num_bins,data_label)
        kde_plot(collated_dict,data_label)
        
        # Used to save the pairwise dicts for the different datasets
        pairwise_saving(collated_data,dataset_names,num_bins=num_bins,ref_index=0)
        pairwise_saving(collated_data,dataset_names,num_bins=num_bins,ref_index=1)

        return dtest, collated_dood, indices_dtest, collated_indices_dood

# One vs Rest mahalanobis
class Mahalanobis_OvR(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)

    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        
        fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))
        
        table_data = {'Class vs Rest': [],'Accuracy': []}

        for i in range(len(np.unique(labels_test))):
            table_data['Class vs Rest'].append(i)
            # Need to copy each time as the values get overwritten
            ovr_labels_test = np.copy(labels_test)
            ovr_indices_dtest = np.copy(indices_dtest)
            
            ovr_labels_test = np.where(ovr_labels_test !=i,-1,ovr_labels_test)
            ovr_labels_test = np.where(ovr_labels_test==i, 0, ovr_labels_test)

            ovr_indices_dtest = np.where(ovr_indices_dtest !=i,-1,ovr_indices_dtest)
            ovr_indices_dtest = np.where(ovr_indices_dtest==i, 0, ovr_indices_dtest)

            
            test_accuracy = self.mahalanobis_classification(ovr_indices_dtest, ovr_labels_test).item()
            table_data['Accuracy'].append(round(test_accuracy,2))

        table_df = pd.DataFrame(table_data)
    
        table = wandb.Table(dataframe=table_df)
        wandb.log({"Mahalanobis One Vs Rest": table})
        table_saving(table_df,'Mahalanobis One vs Rest')
        #import ipdb; ipdb.set_trace()

        return fpr95,auroc,aupr 
          
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        ftrain_norm, ftest_norm, food_norm= self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)

        
        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood
    

class Mahalanobis_OvO(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)

    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        
        column_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        index_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        data = np.zeros((len(index_names),len(column_names)))

        for i in range(len(np.unique(labels_test))):
            for j in range(len(np.unique(labels_test))):
                if i == j: 
                    test_accuracy = torch.tensor(100.0) 
                else:
                    #table_data['Class vs Class'].append(f'{i} vs {j}')
                    train_mask = np.logical_or(labels_train == i, labels_train == j)
                    ovo_labels_train = labels_train[train_mask]
                    max_val = np.amax(ovo_labels_train)
                    ovo_labels_train = ovo_labels_train == max_val # Change to boolean matrix (true and false / 1 and 0s)
                    ovo_features_train =  features_train[train_mask]

                    test_mask = np.logical_or(labels_test ==i, labels_test ==j)
                    ovo_labels_test = labels_test[test_mask]
                    ovo_labels_test = ovo_labels_test == max_val
                    ovo_features_test = features_test[test_mask]
                    ovo_features_ood = features_ood[0:32] # shorten the data of OOD as this is not actually important for the ovo classifier


                    fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
                    np.copy(ovo_features_train),
                    np.copy(ovo_features_test),
                    np.copy(ovo_features_ood),
                    np.copy(ovo_labels_train))

                    test_accuracy = self.mahalanobis_classification(indices_dtest, ovo_labels_test)
                
                data[i,j] = test_accuracy
                #table_data[f'{i}'].append(test_accuracy)
                #table_data['Accuracy'].append(test_accuracy)
        data = np.around(data,decimals=1)

        table_df = pd.DataFrame(data, index = index_names, columns=column_names)
        # plot heat with annotations of the value as well as formating to 2 decimal places          

        # Choose whether to show annotations based on the number of examples present
        if len(np.unique(labels_test)) >10: 
            sns.heatmap(table_df,annot=False,fmt=".1f")
        else:
            sns.heatmap(table_df,annot=True,fmt=".1f")

        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Mahalanobis One vs One Confusion Matrix')
        ovo_filename = f'ovo_conf_matrix.png'
        plt.savefig(ovo_filename,bbox_inches='tight')
        wandb_ovo = f'Mahalanobis One vs One Matrix'    
        wandb.log({wandb_ovo:wandb.Image(ovo_filename)})
        # Update the data table to selectively remove different tables of the data
        row_values = [j for j in range(len(np.unique(labels_test)))]
        updated_data = np.insert(data,0,values = row_values, axis=1)
        column_names.insert(0,'Class') # Inplace operation
        updated_table_df = pd.DataFrame(updated_data, columns=column_names)

        table = wandb.Table(dataframe=updated_table_df)

        wandb.log({"Mahalanobis One Vs One": table})
        # NEED TO CLOSE OTHERWISE WILL HAVE OVERLAPPING MATRICES SAVED IN WANDB
        plt.close()
        return fpr95,auroc,aupr 
    
        
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)

        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood


class Mahalanobis_Subsample(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'fine',
        label_level:str = 'fine',
        quick_callback:bool = True,
        bootstrap_num: int = 10):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)
        
        # Get the number of coarse classes
        self.num_coarse = self.Datamodule.num_coarse_classes
        # Number of times to bootstrap sample the data
        self.bootstrap_num = bootstrap_num

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)

    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        column_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        index_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        data = np.zeros((len(index_names),len(column_names)))

        classes_available = len(np.unique(labels_train))
        accuracy_values = []
        for i in range(self.bootstrap_num):
            specific_classes = np.random.choice(classes_available, size=self.num_coarse, replace=False) # Obtain a sub sample of values
            class_train_masks = [labels_train==specific_class for specific_class in specific_classes] # Obtain masks for class
            train_mask = np.sum(class_train_masks,axis=0) > 0 # Make a joint mask for all te different masks
            specific_labels_train = labels_train[train_mask] # Obtain labels for the different data points
            specific_features_train =  features_train[train_mask] # Nawid - 
            # Test data            
            class_test_masks = [labels_test==specific_class for specific_class in specific_classes]
            test_mask = np.sum(class_test_masks,axis=0) > 0
            specific_labels_test = labels_test[test_mask]
            specific_features_test = features_test[test_mask]

            # Sort the values for the specific class
            sorted_classes = sorted(specific_classes)
            
            features_ood = features_ood[0:32] # shorten the data of OOD as this is not actually important for the ovo classifier
            # Map the subsamples labels to values between 0 and n where n is the number of coarse labels used 
            for j in range(len(specific_classes)):
                #import ipdb; ipdb.set_trace()
                specific_labels_train[specific_labels_train == sorted_classes[j]] = j
                specific_labels_test[specific_labels_test == sorted_classes[j]] = j 
                
            fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
                    np.copy(specific_features_train),
                    np.copy(specific_features_test),
                    np.copy(features_ood),
                    np.copy(specific_labels_train))
            
            

            test_accuracy = self.mahalanobis_classification(indices_dtest, specific_labels_test).item()
            accuracy_values.append(test_accuracy)
        
        
        # Calculate statistics related to the value
        table_data = {'Mean':[],'Std':[],'Min':[], 'Max':[]}
        table_data['Mean'].append(statistics.mean(accuracy_values))
        table_data['Std'].append(statistics.stdev(accuracy_values))
        table_data['Min'].append(min(accuracy_values))
        table_data['Max'].append(max(accuracy_values))

        table_df = pd.DataFrame(table_data)
        
        table_saving(table_df,'Fine Grain Subsampling')
        # NEED TO CLOSE OTHERWISE WILL HAVE OVERLAPPING MATRICES SAVED IN WANDB
        
        
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)

        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood
    

class Class_Mahalanobis_OOD(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,vector_level,label_level,quick_callback)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)
    
    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 

        train_loader = self.Datamodule.train_dataloader()
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


    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        dtest_class = [dtest[indices_dtest==i] for i in np.unique(indices_dtest)]
        dood_class = [dood[indices_dood ==i] for i in np.unique(indices_dtest)] # Make it so that the unique indices are the same for both cases
        # NEED TO MAKE IT SO THAT THE CLASS WISE VALUES CAN BE OBTAINED FOR THE TASK
        
        
        self.AUROC_saving(dtest_class,dood_class,
        f'Class Wise Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC',
        f'Class Wise Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC Table')
        '''
        table_data = {'Class':[], 'AUROC': []}
        
        #import ipdb; ipdb.set_trace()

        for class_num in range(len(dtest_class)):
            if len(dtest_class[class_num]) ==0 or len(dood_class[class_num])==0:
                class_AUROC = -1.0
            else:  
                class_AUROC = get_roc_sklearn(dtest_class[class_num],dood_class[class_num])
            
            table_data['Class'].append(class_num)
            table_data['AUROC'].append(round(class_AUROC,2))

        table_df = pd.DataFrame(table_data)
        print(table_df)
        table = wandb.Table(dataframe=table_df)
        wandb.log({"Class Mahalanobis AUROC":table})
        table_saving(table_df,"Class Mahalanobis AUROC Table")
        '''

    def AUROC_saving(self,class_ID_scores, class_OOD_scores,wandb_name, table_name):
        # NEED TO MAKE IT SO THAT THE CLASS WISE VALUES CAN BE OBTAINED FOR THE TASK
        table_data = {'Class':[], 'AUROC': []}
        
        for class_num in range(len(class_ID_scores)):
            if len(class_ID_scores[class_num]) ==0 or len(class_OOD_scores[class_num])==0:
                class_AUROC = -1.0
            else:  
                class_AUROC = get_roc_sklearn(class_ID_scores[class_num],class_OOD_scores[class_num])
            
            table_data['Class'].append(class_num)
            table_data['AUROC'].append(round(class_AUROC,2))

        table_df = pd.DataFrame(table_data)
        #print(table_df)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)
        



def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

# calculates aupr
def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr

# Nawid - calculate false positive rate
def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

def get_roc_plot(xin, xood,OOD_name):
    anomaly_targets = [0] * len(xin)  + [1] * len(xood)
    outputs = np.concatenate((xin, xood))

    fpr, trp, thresholds = skm.roc_curve(anomaly_targets, outputs)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=fpr, y=trp,
    legend="full",
    alpha=0.3
    )
    # Set  x and y-axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    ROC_filename = f'Images/ROC_{OOD_name}.png'
    plt.savefig(ROC_filename)
    wandb_ROC = f'ROC curve: OOD dataset {OOD_name}'
    wandb.log({wandb_ROC:wandb.Image(ROC_filename)})

    '''
    wandb.log({f'ROC_{OOD_name}': wandb.plot.roc_curve(anomaly_targets, outputs,#scores,
                        labels=None, classes_to_plot=None)})
    '''
    

def count_histogram(input_data,num_bins,name):
    sns.displot(data = input_data,multiple ='stack',stat ='count',common_norm=False, bins=num_bins)#,kde =True)
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    # Used to fix the x limit
    plt.xlim([0, 500])
    plt.title(f'Mahalanobis Distance Counts {name}')
    histogram_filename = f'Images/Mahalanobis_distance_counts_{name}.png'
    plt.savefig(histogram_filename,bbox_inches='tight')  #bbox inches used to make it so that the title can be seen effectively
    plt.close()
    wandb_distance = f'Mahalanobis Distance Counts {name}'
    wandb.log({wandb_distance:wandb.Image(histogram_filename)})
    

def probability_histogram(input_data,num_bins,name):
    sns.displot(data = input_data,multiple ='stack',stat ='probability',common_norm=False, bins=num_bins)#,kde =True)
    plt.xlabel('Distance')
    plt.ylabel('Probability')
    # Used to fix the x limit
    plt.xlim([0, 500])
    plt.ylim([0, 1])
    plt.title(f'Mahalanobis Distance Probabilities {name}')
    histogram_filename = f'Images/Mahalanobis_distances_probabilities_{name}.png'
    plt.savefig(histogram_filename,bbox_inches='tight')  #bbox inches used to make it so that the title can be seen effectively
    plt.close()
    wandb_distance = f'Mahalanobis Distance Probabilities {name}'
    wandb.log({wandb_distance:wandb.Image(histogram_filename)})
    
def kde_plot(input_data,name):
    sns.displot(data =input_data,fill=False,common_norm=False,kind='kde')
    plt.xlabel('Distance')
    plt.ylabel('Normalized Density')
    plt.xlim([0, 500])
    plt.title(f'Mahalanobis Distances {name}')
    kde_filename = f'Images/Mahalanobis_distances_kde_{name}.png'
    plt.savefig(kde_filename,bbox_inches='tight')
    plt.close()
    wandb_distance = f'Mahalanobis Distance KDE {name}'
    wandb.log({wandb_distance:wandb.Image(kde_filename)})
    
def pairwise_saving(collated_data,dataset_names,num_bins,ref_index):
    table_data = {'Dataset':[],'Count Absolute Deviation':[],'Prob Absolute Deviation':[],'KL (Nats)':[], 'JS (Nats)':[],'KS':[]}
    # Calculates the values in a pairwise 
    # Calculate the name of the data based on the ref index
    assert ref_index ==0 or ref_index ==1,"ref index only can be 0 or 1 currently" 
    if ref_index == 0:
        ref = 'Train'
    else:
        ref = 'Test'
    
    for i in range(len(collated_data)-(1+ref_index)):
        pairwise_dict = {}
        # Update the for base case 
        index_val = 1 + i +ref_index
        #import ipdb; ipdb.set_trace()
        pairwise_dict.update({dataset_names[ref_index]:collated_data[ref_index]})
        pairwise_dict.update({dataset_names[index_val]:collated_data[index_val]})
        data_label = f'{ref} Reference - {dataset_names[index_val]}' 
        
        # Plots the counts, probabilities as well as the kde for pairwise
        count_histogram(pairwise_dict,num_bins,data_label)
        probability_histogram(pairwise_dict,num_bins,data_label)
        kde_plot(pairwise_dict,data_label)
                    
        # https://www.kite.com/python/answers/how-to-plot-a-histogram-given-its-bins-in-python 
        # Plots the histogram of the pairwise distance
        count_hist1, _ = np.histogram(pairwise_dict[dataset_names[ref_index]],range=(0,500), bins = num_bins)
        count_hist2, bin_edges = np.histogram(pairwise_dict[dataset_names[index_val]],range=(0,500), bins = num_bins)
        count_absolute_deviation  = np.sum(np.absolute(count_hist1 - count_hist2))

        # Using density =  True is the same as making it so that you normalise each term by the sum of the counts
        prob_hist1, _ = np.histogram(pairwise_dict[dataset_names[ref_index]],range=(0,500), bins = num_bins,density = True)
        prob_hist2, bin_edges = np.histogram(pairwise_dict[dataset_names[index_val]],range=(0,500), bins = num_bins,density= True)
        prob_absolute_deviation  = round(np.sum(np.absolute(prob_hist1 - prob_hist2)),3)
        kl_div = round(kl_divergence(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        js_div = round(js_metric(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        ks_stat = round(ks_statistic_kde(pairwise_dict[dataset_names[ref_index]], pairwise_dict[dataset_names[index_val]]),3)
        
        table_data['Dataset'].append(data_label)
        table_data['Count Absolute Deviation'].append(count_absolute_deviation)
        table_data['Prob Absolute Deviation'].append(prob_absolute_deviation)
        table_data['KL (Nats)'].append(kl_div)
        table_data['JS (Nats)'].append(js_div)
        table_data['KS'].append(ks_stat)
    
    table_df = pd.DataFrame(table_data)
    
    table = wandb.Table(dataframe=table_df)
    wandb.log({f"{ref} Distance statistics": table})
    table_saving(table_df,f'Mahalanobis Distance {ref} Statistics')
    

def table_saving(table_dataframe,name):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    #https://stackoverflow.com/questions/15514005/how-to-change-the-tables-fontsize-with-matplotlib-pyplot
    data_table = ax.table(cellText=table_dataframe.values, colLabels=table_dataframe.columns, loc='center')
    data_table.set_fontsize(24)
    data_table.scale(2.0, 2.0)  # may help
    filename = name.replace(" ","_") # Change the values with the empty space to underscore
    filename = f'Images/{filename}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    wandb_title = f'{name}'
    #import ipdb; ipdb.set_trace()
    wandb.log({wandb_title:wandb.Image(filename)})

