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

from Contrastive_uncertainty.toy_replica.toy_general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.toy_replica.toy_general.utils.pl_metrics import precision_at_k, mean


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
        
        # Number of classes of the data
        # Names for creating a confusion matrix for the data
        class_dict = self.Datamodule.idx2class
        self.class_names = [v for k,v in class_dict.items()] # names of the categories of the dataset
        # Obtain class names list for the case of the OOD data
        OOD_class_dict = self.OOD_Datamodule.idx2class
        self.OOD_class_names = [v for k,v in OOD_class_dict.items()] # names of the categories of the dataset
        #import ipdb; ipdb.set_trace()
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
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        num_classes = max(labels_train+1)
        fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            num_classes)

        self.mahalanobis_classification(indices_dtest, labels_test,f'Mahalanobis Classification: {self.vector_level}: {self.label_level}')

        # Calculates the mahalanobis distance using unsupervised approach 
        return fpr95,auroc,aupr 

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels,name):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        #import ipdb; ipdb.set_trace()       
        wandb.log({name:mahalanobis_test_accuracy})

    def get_features(self, pl_module, dataloader, max_images=10**10, verbose=False):
        features, labels = [], []
        
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
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
                
            if total > max_images:
                break
            
            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][self.vector_level](img) # Performs the callback for the desired level
            
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
            
            if verbose and not index % 50:
                print(index)
                
            total += len(img)  
        
        return np.array(features), np.array(labels)
    
    # Nawid - perform k-means on the training features and then assign a cluster assignment
    def get_clusters(self, ftrain, nclusters):
        kmeans = faiss.Kmeans(
            ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=True
        )
        kmeans.train(np.random.permutation(ftrain))
        _, ypred = kmeans.assign(ftrain)
        return ypred
    
    def get_scores(self,ftrain, ftest, food, labelstrain,n_clusters):
        if labelstrain is None:
            ypred = self.get_clusters(ftrain=ftrain, nclusters=n_clusters)
        else:
            ypred = labelstrain
        return self.get_scores_multi_cluster(ftrain, ftest, food, ypred)
    
    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
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
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain,num_clusters)
        self.log_confidence_scores(dtest,dood,labelstrain,num_clusters)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
        
        
        wandb.log({self.log_name + f' AUROC: {self.vector_level} vector: {num_clusters} classes: {self.OOD_dataname}': auroc})
        return fpr95, auroc, aupr, indices_dtest, indices_dood
    
    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,Dtest,DOOD,labels_train,num_clusters):  
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