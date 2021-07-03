import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random

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

# Callback to calculate scores for a particular level, not for the hierarchical case
class scores_comparison(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__(Datamodule,OOD_Datamodule,vector_level, label_level, quick_callback)
        #print('General scores being used')  
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        pass
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self, trainer, pl_module):
        #print('General scores being used') 
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level, self.label_level)

        dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train))

        
        self.class_specific_scores(dtest,indices_dtest,'ID',f'Class data scores ID {self.vector_level} {self.label_level}')
        self.class_specific_scores(dood,indices_dood,'OOD',f'Class data scores OOD {self.vector_level} {self.label_level} {self.OOD_dataname} ')
        self.ID_OOD_scores(dtest,dood)
        

    # Mahalanobis distance scores for a particular class
    def class_specific_scores(self,ddata, indices,data_distribution,class_data_name):
        ddata_class = [pd.DataFrame(ddata[indices == i]) for i in np.unique(indices)] # Nawid - training data which have been predicted to belong to a particular class
        
        # Concatenate all the dataframes (which places nans in situations where the columns have different lengths)
        class_table_df = pd.concat(ddata_class,axis=1)
        #https://stackoverflow.com/questions/30647247/replace-nan-in-a-dataframe-with-random-values
        class_table_df = class_table_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        #class_table_df = class_table_df.fillna(-1) # replace nans with -1
        class_table_df.columns =[f'{data_distribution} {self.vector_level} {self.label_level} class {i}' for i in np.unique(indices)]
        
        class_table = wandb.Table(data=class_table_df)
        wandb.log({class_data_name:class_table})
    
    # Mahalanobis distance scores for the ID and OOD data
    def ID_OOD_scores(self,din,dood):
        limit = min(len(din),len(dood))
        din = din[:limit]
        dood = dood[:limit] 
        # Actual code fort the normal case
        # https://towardsdatascience.com/merge-dictionaries-in-python-d4e9ce137374
        #all_dict = {**ID_dict,**OOD_dict} # Merged dictionary
        data_dict = {f'ID {self.vector_level} {self.label_level}': din, f'{self.OOD_dataname} {self.vector_level} {self.label_level}':dood}
        # Plots the counts, probabilities as well as the kde
        data_name = f'{self.vector_level} {self.label_level} {self.OOD_dataname} data scores'
        
        table_df = pd.DataFrame(data_dict)
        table = wandb.Table(data=table_df)
        wandb.log({data_name:table})

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
        #import ipdb; ipdb.set_trace()              

        return np.array(features), np.array(labels)

    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        return dtest, dood, indices_dtest, indices_dood
