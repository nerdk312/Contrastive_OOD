from numpy.lib.financial import _ipmt_dispatcher
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



class Relative_Mahalanobis(Mahalanobis_OOD):
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
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level, self.label_level)

        dtest, dood, indices_dtest, indices_dood = self.get_eval_results(features_train, features_test, features_ood, labels_train)
        # Calculate AUROC
        auroc = get_roc_sklearn(dtest, dood)
        wandb.run.summary[f'Relative Mahalanobis AUROC: {self.vector_level} vector: {self.label_level} labels : {self.OOD_dataname}'] = auroc

        # Saves the confidence valeus of the data table
        limit = min(len(dtest),len(dood))
        dtest = dtest[:limit]
        dood = dood[:limit]

        # https://towardsdatascience.com/merge-dictionaries-in-python-d4e9ce137374
        #all_dict = {**ID_dict,**OOD_dict} # Merged dictionary
        data_dict = {f'ID {self.vector_level} {self.label_level}': dtest, f'{self.OOD_dataname} {self.vector_level} {self.label_level}':dood}
        # Plots the counts, probabilities as well as the kde
        data_name = f'Relative Mahalanobis - {self.vector_level} - {self.label_level} - {self.OOD_dataname} data scores'
        
        table_df = pd.DataFrame(data_dict)
        table = wandb.Table(data=table_df)

        wandb.log({data_name:table})


        relative_mahalanobis_test_accuracy = self.mahalanobis_classification(indices_dtest, labels_test)
        wandb.run.summary[f'Relative Mahalanobis Classification: {self.vector_level}: {self.label_level}'] = relative_mahalanobis_test_accuracy
    
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
        # Calculate the mean of the entire diataset and the covariance of the entire training set
        mean = np.mean(ftrain, axis=0,keepdims=True)
        cov = np.cov(ftrain.T, bias=True)
        
        background_din = np.sum(
                ftest - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftest- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        background_dood = np.sum(
                food - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (food- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        din = din - background_din
        dood = dood - background_dood

        '''
        # Checks how the code works for the task
        #import ipdb; ipdb.set_trace()
        array = []
        array1 = np.array([3,4])
        array2 = np.array([1,2])
        array3 = np.array([4,7])
        array.append(array1)
        array.append(array2)
        array.append(array3)
        background_val = np.array([-1,10])
        import ipdb; ipdb.set_trace()
        '''

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


# Obtain the relative mahalanobis scores for the class specific case
class Class_Relative_Mahalanobis(Relative_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level, label_level, quick_callback)

    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)


    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level, self.label_level)


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
        
        # Uses the relative mahalanobis approach to obtain the scores for the data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)

        self.AUROC_saving(dtest, indices_dtest,
            dood,indices_dood,labelstrain,
            f'Class Wise Relative Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC',
            f'Class Wise Relative Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC Table')
    
    def AUROC_saving(self,ID_scores,indices_ID, OOD_scores, indices_OOD,labels,wandb_name, table_name):
        # NEED TO MAKE IT SO THAT THE CLASS WISE VALUES CAN BE OBTAINED FOR THE TASK as well as the fraction of data points in a particular class
        table_data = {'Class':[], 'AUROC': [], 'ID Samples Fraction':[], 'OOD Samples Fraction':[]}
        #np.unique(indices_ID)
        class_ID_scores = [ID_scores[indices_ID==i] for i in np.unique(labels)]
        class_OOD_scores = [OOD_scores[indices_OOD==i] for i in np.unique(labels)]
    

        for class_num in range(len(np.unique(labels))):
            if len(class_ID_scores[class_num]) ==0 or len(class_OOD_scores[class_num])==0:
                class_AUROC = -1.0
            else:  
                class_AUROC = get_roc_sklearn(class_ID_scores[class_num],class_OOD_scores[class_num])
            
            class_ID_fraction = len(class_ID_scores[class_num])/len(ID_scores)
            class_OOD_fraction = len(class_OOD_scores[class_num])/len(OOD_scores)
            table_data['Class'].append(f'{class_num}')
            table_data['AUROC'].append(round(class_AUROC,2))
            table_data['ID Samples Fraction'].append(round(class_ID_fraction,2))
            table_data['OOD Samples Fraction'].append(round(class_OOD_fraction,2))
   
        # calculate the AUROC for the dataset in general
        All_AUROC = get_roc_sklearn(ID_scores,OOD_scores)
        #table_data['Class'].append(-1)
        table_data['Class'].append('All')
        table_data['AUROC'].append(round(All_AUROC,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)

        table_df = pd.DataFrame(table_data)
        #print(table_df)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)

# Uses the inverted relative mahalanobis distance instead of the usual mahalanobis distance to calculate the scores, only difference is changing the subtraction to a plus in the get scores function
class Class_Inverted_Relative_Mahalanobis(Class_Relative_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level, label_level, quick_callback)

    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)


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
        # Calculate the mean of the entire diataset and the covariance of the entire training set
        mean = np.mean(ftrain, axis=0,keepdims=True)
        cov = np.cov(ftrain.T, bias=True)
        
        background_din = np.sum(
                ftest - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftest- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        background_dood = np.sum(
                food - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (food- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        din = din + background_din
        dood = dood + background_dood

        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        ftrain_norm,ftest_norm,food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        
        # Uses the relative mahalanobis approach to obtain the scores for the data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)

        self.AUROC_saving(dtest, indices_dtest,
            dood,indices_dood,labelstrain,
            f'Class Wise Inverted Relative Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC',
            f'Class Wise Inverted Relative Mahalanobis {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC Table')
    
    