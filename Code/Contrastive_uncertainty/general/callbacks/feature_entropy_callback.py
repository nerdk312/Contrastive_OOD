from numpy.core.numeric import indices
from numpy.lib.ufunclike import isposinf
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



class Feature_Entropy(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        self.OOD_dataname = self.OOD_Datamodule.name

        self.vector_level = 'instance'
        self.label_level = 'fine'


    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    
    # Performs all the computation in the callback
    def forward_callback(self, trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 


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
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        self.get_eval_results(
            np.copy(features_train),
            np.copy(labels_train))
            
    

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
    
    def normalise(self,ftrain):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        
        return ftrain
    
    
    def get_marginal_entropy(self, ftrain, marginal_wandb_name):
        # Calculate the mean along the feature dimension
        feature_mean = np.mean(ftrain,axis=0)
        feature_std = np.std(ftrain,axis=0)
        
        marginal_feature_entropy = {'Dimension': [], 'Entropy (Nats)': []}
        
        for i in range(len(feature_mean)):
            m = torch.distributions.normal.Normal(torch.tensor([feature_mean[i]]),torch.tensor([feature_std[i]])) 
            entropy_value = m.entropy().item() # Get as a scalar rather than tensor oject
            marginal_feature_entropy['Dimension'].append(i)
            marginal_feature_entropy['Entropy (Nats)'].append(round(entropy_value,3))

        table_df = pd.DataFrame(marginal_feature_entropy)
        marginal_entropy_table = wandb.Table(dataframe = table_df)
        
        wandb.log({marginal_wandb_name: marginal_entropy_table})

    def get_conditional_entropy(self, ftrain, labelstrain, conditional_wandb_name):
        xc = [ftrain[labelstrain==i] for i in np.unique(labelstrain)]
        class_feature_means = [np.mean(xc[i],axis=0) for i in np.unique(labelstrain)]
        class_feature_std = [np.std(xc[i],axis=0) for i in np.unique(labelstrain)]
        dimensionality = len(class_feature_means[0])
        num_classes = len(xc)

        conditional_feature_entropy = {'Dimension':[], 'Entropy (Nats)': []}
        # Go through the different dimensions of the data
        for i in range(dimensionality):
            # Go through the different classes
            all_class_specific_entropy = []
            for j in range(num_classes):
                m = torch.distributions.normal.Normal(torch.tensor([class_feature_means[j][i]]),torch.tensor([class_feature_std[j][i]]))
                class_specific_entropy = m.entropy().item()
                all_class_specific_entropy.append(class_specific_entropy)
            
            average_class_specific_entropy = np.mean(all_class_specific_entropy)
            conditional_feature_entropy['Dimension'].append(i)
            conditional_feature_entropy['Entropy (Nats)'].append(round(average_class_specific_entropy,3))

        table_df = pd.DataFrame(conditional_feature_entropy)
        conditional_entropy_table = wandb.Table(dataframe = table_df)
        
        wandb.log({conditional_wandb_name: conditional_entropy_table})



    def get_eval_results(self, ftrain, labelstrain):
        ftrain_norm = self.normalise(ftrain)
        self.get_marginal_entropy(ftrain,'Marginal Feature Entropy')
        self.get_conditional_entropy(ftrain,labelstrain, 'Class Conditional Feature Entropy')
    

        
