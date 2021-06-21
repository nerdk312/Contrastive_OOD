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
import wandb
import sklearn.metrics as skm
import faiss

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean

from Contrastive_uncertainty.general.callbacks.compare_distributions import ks_statistic, ks_statistic_kde, js_metric, kl_divergence




class Aggregated_Mahalanobis_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):
        #import ipdb; ipdb.set_trace()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        self.log_name = "Mahalanobis"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        
        
        self.OOD_dataname = self.OOD_Datamodule.name
        

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train_instance, _ = self.get_features(pl_module, train_loader,'instance','fine')
        features_train_fine, labels_train_fine = self.get_features(pl_module, train_loader,'fine','fine')
        features_train_coarse, labels_train_coarse = self.get_features(pl_module, train_loader,'coarse','coarse')
        
        features_test_instance, _ = self.get_features(pl_module, test_loader,'instance','fine')
        features_test_fine, labels_test_fine = self.get_features(pl_module, test_loader,'fine','fine') 
        features_test_coarse, labels_test_coarse = self.get_features(pl_module, test_loader,'coarse','coarse')

        features_ood_instance, _ = self.get_features(pl_module, ood_loader,'instance','fine')
        features_ood_fine, labels_ood_fine = self.get_features(pl_module, ood_loader,'fine','fine')
        features_ood_coarse, labels_ood_coarse = self.get_features(pl_module, ood_loader,'coarse','coarse')

        features_train = [features_train_instance, features_train_fine, features_train_coarse]
        features_test = [features_test_instance, features_test_fine, features_test_coarse]
        features_ood = [features_ood_instance, features_ood_fine, features_ood_coarse]
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        num_classes = max(labels_train_fine + 1)
        fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train_fine),
            num_classes)

        # Calculates the mahalanobis distance using unsupervised approach 
        return fpr95,auroc,aupr 

    def get_features(self, pl_module, dataloader,vector_level, label_level, max_images=10**10, verbose=False):
        features, labels = [], []
        
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, *label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][label_level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
                
            if total > max_images:
                break
            
            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][vector_level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
            
            if verbose and not index % 50:
                print(index)
                
            total += len(img)  
        
        return np.array(features), np.array(labels)
    
    def get_scores(self,ftrain, ftest, food, labelstrain,n_clusters):
        ypred = labelstrain
        return self.get_scores_multi_cluster(ftrain, ftest, food, ypred)

    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        total_din = []
        total_dood = []
        for index in range(3):
            xc = [ftrain[index][ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
            din = [
                np.sum(
                    (ftest[index] - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                    * (
                        np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                            (ftest[index] - np.mean(x, axis=0, keepdims=True)).T
                        ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                    ).T,
                    axis=-1,
                )
                for x in xc # Nawid - done for all the different classes
            ]
        
            dood = [
                np.sum(
                    (food[index] - np.mean(x, axis=0, keepdims=True))
                    * (
                        np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                            (food[index] - np.mean(x, axis=0, keepdims=True)).T
                        )
                    ).T,
                    axis=-1,
                )
                for x in xc # Nawid- this calculates the score for all the OOD examples 
            ]
            #import ipdb; ipdb.set_trace()
            # total is a list of lists which takes in a list din (or dood)
            total_din.append(din)
            total_dood.append(dood)

        # import ipdb; ipdb.set_trace()       
        # aggregated sums together the values in the list of lists into a single list https://stackoverflow.com/questions/14050824/add-sum-of-values-of-two-lists-into-new-list
        aggregated_din = [sum(x) for x in zip(*total_din)]
        aggregated_dood = [sum(x) for x in zip(*total_dood)]
        
        
            #total_din += din
            #total_dood +=dood
        
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(aggregated_din,axis = 0)
        indices_dood = np.argmin(aggregated_dood, axis=0)
                 
        aggregated_din = np.min(aggregated_din, axis=0) # Nawid - calculate the minimum distance 
        aggregated_dood = np.min(aggregated_dood, axis=0)

        collated_din_values = []
        collated_dood_values = []

        for index in range(3):
            # Concatenate the different classes into a single array
            single_branch_in, single_branch_ood = torch.from_numpy(np.column_stack(total_din[index])), torch.from_numpy(np.column_stack(total_dood[index]))
            
            # Make into a tensor and make it have the same shape as the other case
            tensor_indices_din, tensor_indices_dood = torch.from_numpy(indices_din).unsqueeze(1), torch.from_numpy(indices_dood).unsqueeze(1)
            
            # Obtain the values corresponding to the prediction indices
            specific_values_in, specific_values_ood = torch.gather(single_branch_in,1,tensor_indices_din), torch.gather(single_branch_ood,1,tensor_indices_dood)
            collated_din_values.append(specific_values_in)
            collated_dood_values.append(specific_values_ood)

        
        # Scores for the particular case where there are different values present
        collated_in_scores, collated_ood_scores = torch.cat(collated_din_values,dim=1), torch.cat(collated_dood_values,dim=1)

        # Characteristic properties of the data for the task
        mean_collated_in_scores, mean_collated_ood_scores = torch.mean(collated_in_scores,dim=1),torch.mean(collated_ood_scores,dim=1)
        std_collated_in_scores, std_collated_ood_scores = torch.std(collated_in_scores,dim=1), torch.std(collated_ood_scores,dim=1)
        min_collated_in_scores, _ = torch.min(collated_in_scores,dim=1)
        max_collated_in_scores,_ = torch.max(collated_in_scores,dim=1)
        
        min_collated_ood_scores, _ = torch.min(collated_ood_scores,dim=1)
        max_collated_ood_scores,_ = torch.max(collated_ood_scores,dim=1)
        #import ipdb; ipdb.set_trace()
        in_statistics = (mean_collated_in_scores,std_collated_in_scores,min_collated_in_scores,max_collated_in_scores)
        ood_statistics = (mean_collated_ood_scores,std_collated_ood_scores,min_collated_ood_scores,max_collated_ood_scores)

        return aggregated_din, aggregated_dood, in_statistics, ood_statistics, indices_din, indices_dood
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain, num_clusters):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        for index in range(3):
            
            ftrain[index] /= np.linalg.norm(ftrain[index], axis=-1, keepdims=True) + 1e-10
            ftest[index] /= np.linalg.norm(ftest[index], axis=-1, keepdims=True) + 1e-10
            food[index] /= np.linalg.norm(food[index], axis=-1, keepdims=True) + 1e-10
            # Nawid - calculate the mean and std of the traiing features
            m, s = np.mean(ftrain[index], axis=0, keepdims=True), np.std(ftrain[index], axis=0, keepdims=True)
            # Nawid - normalise data using the mean and std
            ftrain[index] = (ftrain[index] - m) / (s + 1e-10)
            ftest[index] = (ftest[index] - m) / (s + 1e-10)
            food[index] = (food[index] - m) / (s + 1e-10)
            # Nawid - obtain the scores for the test data and the OOD data
        aggregated_dtest, aggregated_dood, in_statistics, ood_statistics,indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain, num_clusters)
                
        self.log_confidence_scores(in_statistics,ood_statistics,num_clusters)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(aggregated_dtest, aggregated_dood)

        # Calculates the AUROC
        auroc, aupr = get_roc_sklearn(aggregated_dtest, aggregated_dood), get_pr_sklearn(aggregated_dtest, aggregated_dood)
        
        wandb.log({self.log_name + f' Aggregated AUROC: {num_clusters} classes: {self.OOD_dataname}': auroc})
        wandb.log({self.log_name + f' Aggregated AUPR: {num_clusters} classes: {self.OOD_dataname}': aupr})
        wandb.log({self.log_name + f' Aggregated FPR: {num_clusters} classes: {self.OOD_dataname}': fpr95})
        
        return fpr95, auroc, aupr, indices_dtest, indices_dood
    
    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,in_statistics,ood_statistics,num_clusters):  
        names = ['Mean','Std','Min','Max']
        # confidence_test_mean, confidence_test_std, confidence_test_min, confidence_test_max = in_statistics
        # confidence_OOD_mean, confidence_OOD_std, confidence_OOD_min, confidence_OOD_max = ood_statistics

        for index in range(len(in_statistics)):
            true_data = [[s] for s in in_statistics[index]]
            true_table = wandb.Table(data=true_data, columns=["scores"])

            true_histogram_name = self.true_histogram + f': Aggregated {names[index]}: {num_clusters} classes'
            ood_histogram_name = self.ood_histogram + f': Aggregated {names[index]}: {num_clusters} classes: {self.OOD_dataname}'

            wandb.log({true_histogram_name: wandb.plot.histogram(true_table, "scores",title=true_histogram_name)})

            # Histogram of the confidence scores for the OOD data
            ood_data = [[s] for s in ood_statistics[index]]
            ood_table = wandb.Table(data=ood_data, columns=["scores"])
            wandb.log({ood_histogram_name: wandb.plot.histogram(ood_table, "scores",title=ood_histogram_name)})
        
         



class Differing_Mahalanobis_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):
        #import ipdb; ipdb.set_trace()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        self.log_name = "Mahalanobis"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train_instance, _ = self.get_features(pl_module, train_loader,'instance','fine')
        features_train_fine, labels_train_fine = self.get_features(pl_module, train_loader,'fine','fine')
        features_train_coarse, labels_train_coarse = self.get_features(pl_module, train_loader,'coarse','coarse')
        
        features_test_instance, _ = self.get_features(pl_module, test_loader,'instance','fine')
        features_test_fine, labels_test_fine = self.get_features(pl_module, test_loader,'fine','fine') 
        features_test_coarse, labels_test_coarse = self.get_features(pl_module, test_loader,'coarse','coarse')

        features_ood_instance, _ = self.get_features(pl_module, ood_loader,'instance','fine')
        features_ood_fine, labels_ood_fine = self.get_features(pl_module, ood_loader,'fine','fine')
        features_ood_coarse, labels_ood_coarse = self.get_features(pl_module, ood_loader,'coarse','coarse')

        features_train = [features_train_instance, features_train_fine, features_train_coarse]
        # Using the fine and coarse labels to obtain the OOD scores in this case
        labels_train = [labels_train_fine,labels_train_fine, labels_train_coarse]

        features_test = [features_test_instance, features_test_fine, features_test_coarse]
        features_ood = [features_ood_instance, features_ood_fine, features_ood_coarse]
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        num_classes = max(labels_train_fine + 1)
        fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            num_classes)

        # Calculates the mahalanobis distance using unsupervised approach 
        return fpr95, auroc, aupr, indices_dtest, indices_dood

    def get_features(self, pl_module, dataloader,vector_level, label_level, max_images=10**10, verbose=False):
        features, labels = [], []
        
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, *label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][label_level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
                
            if total > max_images:
                break
            
            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][vector_level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
            
            if verbose and not index % 50:
                print(index)
                
            total += len(img)  
        
        return np.array(features), np.array(labels)
    
    def get_scores(self,ftrain, ftest, food, labelstrain,n_clusters):
        ypred = labelstrain
        return self.get_scores_multi_cluster(ftrain, ftest, food, ypred)

    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        total_din = []
        total_dood = []
        for index in range(3):
            xc = [ftrain[index][ypred[index] == i] for i in np.unique(ypred[index])] # Nawid - training data which have been predicted to belong to a particular class
        
            din = [
                np.sum(
                    (ftest[index] - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                    * (
                        np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                            (ftest[index] - np.mean(x, axis=0, keepdims=True)).T
                        ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                    ).T,
                    axis=-1,
                )
                for x in xc # Nawid - done for all the different classes
            ]
        
            dood = [
                np.sum(
                    (food[index] - np.mean(x, axis=0, keepdims=True))
                    * (
                        np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                            (food[index] - np.mean(x, axis=0, keepdims=True)).T
                        )
                    ).T,
                    axis=-1,
                )
                for x in xc # Nawid- this calculates the score for all the OOD examples 
            ]
            # Inverse mapping for the coarse situation
            if index ==2:
                # Make new lists to contain values from the mapping
                mapped_din = []
                mapped_dood = []
                # Go through the coarse mapping
                
                for j,coarse in enumerate(self.Datamodule.coarse_mapping):
                    #import ipdb; ipdb.set_trace()
                    mapped_din.insert(j,din[coarse])
                    mapped_dood.insert(j,dood[coarse])
                    #mapped_din[j] =  din[coarse]
                    #mapped_dood[j] = dood[coarse]
                # Update the values with the coarse to fine
                #import ipdb; ipdb.set_trace()
                din = mapped_din
                dood = mapped_dood

            # total is a list of lists which takes in a list din (or dood)
            total_din.append(din)
            total_dood.append(dood)

        # import ipdb; ipdb.set_trace()       
        # aggregated sums together the values in the list of lists into a single list https://stackoverflow.com/questions/14050824/add-sum-of-values-of-two-lists-into-new-list
        aggregated_din = [sum(x) for x in zip(*total_din)]
        aggregated_dood = [sum(x) for x in zip(*total_dood)]
        
            #total_din += din
            #total_dood +=dood
        
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(aggregated_din,axis = 0)
        indices_dood = np.argmin(aggregated_dood, axis=0)
                 
        aggregated_din = np.min(aggregated_din, axis=0) # Nawid - calculate the minimum distance 
        aggregated_dood = np.min(aggregated_dood, axis=0)

        collated_din_values = []
        collated_dood_values = []

        for index in range(3):
            # Concatenate the different classes into a single array
            single_branch_in, single_branch_ood = torch.from_numpy(np.column_stack(total_din[index])), torch.from_numpy(np.column_stack(total_dood[index]))
            
            # Make into a tensor and make it have the same shape as the other case
            tensor_indices_din, tensor_indices_dood = torch.from_numpy(indices_din).unsqueeze(1), torch.from_numpy(indices_dood).unsqueeze(1)
            
            # Obtain the values corresponding to the prediction indices
            specific_values_in, specific_values_ood = torch.gather(single_branch_in,1,tensor_indices_din), torch.gather(single_branch_ood,1,tensor_indices_dood)
            collated_din_values.append(specific_values_in)
            collated_dood_values.append(specific_values_ood)

        
        # Scores for the particular case where there are different values present
        collated_in_scores, collated_ood_scores = torch.cat(collated_din_values,dim=1), torch.cat(collated_dood_values,dim=1)

        # Characteristic properties of the data for the task
        mean_collated_in_scores, mean_collated_ood_scores = torch.mean(collated_in_scores,dim=1),torch.mean(collated_ood_scores,dim=1)
        std_collated_in_scores, std_collated_ood_scores = torch.std(collated_in_scores,dim=1), torch.std(collated_ood_scores,dim=1)
        min_collated_in_scores, _ = torch.min(collated_in_scores,dim=1)
        max_collated_in_scores,_ = torch.max(collated_in_scores,dim=1)
        
        min_collated_ood_scores, _ = torch.min(collated_ood_scores,dim=1)
        max_collated_ood_scores,_ = torch.max(collated_ood_scores,dim=1)
        #import ipdb; ipdb.set_trace()
        in_statistics = (mean_collated_in_scores,std_collated_in_scores,min_collated_in_scores,max_collated_in_scores)
        ood_statistics = (mean_collated_ood_scores,std_collated_ood_scores,min_collated_ood_scores,max_collated_ood_scores)

        return aggregated_din, aggregated_dood, in_statistics, ood_statistics, indices_din, indices_dood
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain, num_clusters):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        for index in range(3):
            
            ftrain[index] /= np.linalg.norm(ftrain[index], axis=-1, keepdims=True) + 1e-10
            ftest[index] /= np.linalg.norm(ftest[index], axis=-1, keepdims=True) + 1e-10
            food[index] /= np.linalg.norm(food[index], axis=-1, keepdims=True) + 1e-10
            # Nawid - calculate the mean and std of the traiing features
            m, s = np.mean(ftrain[index], axis=0, keepdims=True), np.std(ftrain[index], axis=0, keepdims=True)
            # Nawid - normalise data using the mean and std
            ftrain[index] = (ftrain[index] - m) / (s + 1e-10)
            ftest[index] = (ftest[index] - m) / (s + 1e-10)
            food[index] = (food[index] - m) / (s + 1e-10)
            # Nawid - obtain the scores for the test data and the OOD data
        aggregated_dtest, aggregated_dood, in_statistics, ood_statistics,indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain, num_clusters)
                
        self.log_confidence_scores(in_statistics,ood_statistics,num_clusters)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(aggregated_dtest, aggregated_dood)

        # Calculates the AUROC
        auroc, aupr = get_roc_sklearn(aggregated_dtest, aggregated_dood), get_pr_sklearn(aggregated_dtest, aggregated_dood)
        
        wandb.log({self.log_name + f' Differing AUROC: {num_clusters} classes: {self.OOD_dataname}': auroc})
        wandb.log({self.log_name + f' Differing AUPR: {num_clusters} classes: {self.OOD_dataname}': aupr})
        wandb.log({self.log_name + f' Differing FPR: {num_clusters} classes: {self.OOD_dataname}': fpr95})
        
        return fpr95, auroc, aupr, indices_dtest, indices_dood
    
    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,in_statistics,ood_statistics,num_clusters):  
        names = ['Mean','Std','Min','Max']
        # confidence_test_mean, confidence_test_std, confidence_test_min, confidence_test_max = in_statistics
        # confidence_OOD_mean, confidence_OOD_std, confidence_OOD_min, confidence_OOD_max = ood_statistics

        for index in range(len(in_statistics)):
            true_data = [[s] for s in in_statistics[index]]
            true_table = wandb.Table(data=true_data, columns=["scores"])

            true_histogram_name = self.true_histogram + f': Differing {names[index]}: {num_clusters} classes'
            ood_histogram_name = self.ood_histogram + f': Differing {names[index]}: {num_clusters} classes: {self.OOD_dataname}'

            wandb.log({true_histogram_name: wandb.plot.histogram(true_table, "scores",title=true_histogram_name)})

            # Histogram of the confidence scores for the OOD data
            ood_data = [[s] for s in ood_statistics[index]]
            ood_table = wandb.Table(data=ood_data, columns=["scores"])
            wandb.log({ood_histogram_name: wandb.plot.histogram(ood_table, "scores",title=ood_histogram_name)})