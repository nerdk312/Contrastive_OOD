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




class Dataset_class_variance(pl.Callback):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True, vector_level='fine',label_level='fine'):
    
        super().__init__()
        #import ipdb; ipdb.set_trace()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
   
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name

        self.vector_level = vector_level
        self.label_level = label_level

    def on_test_epoch_end(self, trainer, pl_module):
        #import ipdb; ipdb.set_trace()
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level)
        self.get_eval_results(features_train,labels_train)

    
    def get_features(self, pl_module, dataloader, level):
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

    
    def get_variance(self, ftrain, ypred): # Add additional variables for the parent index
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        class_centroids = [np.mean(x,axis=0,keepdims=True) for x in xc]
        diff = [np.abs(xc[class_num] - class_centroids[class_num]) for class_num in range(len(class_centroids))]
        class_variance_approach1 = [np.mean(val**2) for val in diff]
        

        class_variance_approach2 = [np.var(ftrain[ypred==i]) for i in np.unique(ypred)]
        return class_variance_approach1, class_variance_approach2
    
    def normalise(self,ftrain):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        
        return ftrain
    

    def data_saving(self,class_variance_scores1,class_variance_scores2, wandb_name,table_name):
        '''
        args:
            class_variance_scores : List which contains the class variance for the different classes
            wandb name: name of data table
            table_name: name of the saved table
        '''

        table_data = {'Class':[], 'Variance Approach 1':[],'Variance Approach 2':[]}
        
        # Save variance score for the
        for class_num in range(len(class_variance_scores1)):    
            table_data['Class'].append(f'Class {class_num}')
            score1,score2 =  round(class_variance_scores1[class_num],3), round(class_variance_scores2[class_num],3)
            table_data['Variance Approach 1'].append(score1)
            table_data['Variance Approach 2'].append(score2)        

        # Table saving
        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)
    
    def get_eval_results(self,ftrain,labelstrain):
        ftrain_norm = self.normalise(ftrain)
        class_var1, class_var2 = self.get_variance(ftrain_norm,labelstrain)
        #class_var = [np.var(ftrain_norm[labelstrain==i]) for i in np.unique(labelstrain)]
        self.data_saving(class_var1,class_var2,f'Class Variance {self.vector_level} {self.label_level}',f'Class Variance {self.vector_level} {self.label_level} Table')