#import ipdb
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
        bootstrap_num: int = 50,
        typicality_bsz:int = 25):
        
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

        self.bootstrap_num = bootstrap_num
        self.typicality_bsz = typicality_bsz


    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    
    # Performs all the computation in the callback
    def forward_callback(self, trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        val_loader = self.Datamodule.val_dataloader()
        # Use the test transform validataion loader
        
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader)
        features_val, labels_val = self.get_features(pl_module,val_loader)
        features_test, labels_test = self.get_features(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_val),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_val),
            np.copy(labels_test))
            


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

        
    # General function to ge the thresholds
    def get_class_thresholds(self,fdata,class_means,class_cov,class_entropy,bsz):
        
        if bsz == 1:
            class_thresholds = self.get_class_thresholds_single(fdata,class_means,class_cov,class_entropy)
        else:
            class_thresholds = self.get_class_threshold_batch(fdata,class_means,class_cov,class_entropy,bsz)

        return class_thresholds
    
    # Code to calculate the thresholds in singles
    def get_class_thresholds_single(self,fdata,class_means,class_cov,class_entropy):
        ddata = np.sum(
            (fdata - class_means) # Nawid - distance between the data point and the mean
            * (
                np.linalg.pinv(class_cov).dot(
                    (fdata - class_means).T
                ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
            ).T,
            axis=-1)
        
        nll = - (0.5*(ddata**2))
        threshold_k = np.abs(nll- class_entropy)
        class_thresholds = threshold_k.tolist()
        return class_thresholds

    # Code to calculate the thresholds in batch
    def get_class_threshold_batch(self,fdata,class_means,class_cov,class_entropy,bsz):
        class_thresholds = [] #  List of class threshold values
        # obtain the num batches
        num_batches = len(fdata)//bsz 
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*bsz):((i+1)*bsz)]
            ddata = np.sum(
            (fdata_batch - class_means) # Nawid - distance between the data point and the mean
            * (
                np.linalg.pinv(class_cov).dot(
                    (fdata_batch - class_means).T
                ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
            ).T,
            axis=-1)

            nll = - np.mean(0.5*(ddata**2))
            threshold_k = np.abs(nll- class_entropy)
            class_thresholds.append(threshold_k)
        
        return class_thresholds
    
    def get_online_test_thresholds(self, means, cov, entropy, ftest, ytest,bsz):
        test_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features
        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]
        # Iterate through the classes
        for class_num in range(len(np.unique(ytest))):
            # Get data for a particular class
            xtest_class = xtest_c[class_num] 
            class_test_thresholds = self.get_class_thresholds(xtest_class, means[class_num], cov[class_num],entropy[class_num],bsz) # Get class thresholds for the particular class
            test_thresholds.append(class_test_thresholds)

        return test_thresholds
    
    def get_online_ood_thresholds(self, means, cov, entropy, food,bsz):
        # Treating other classes as OOD dataset for a particular class
        ood_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features

        # Iterate through the classes
        for class_num in range(len(means)):
            class_ood_thresholds = self.get_class_thresholds(food,means[class_num], cov[class_num],entropy[class_num],bsz)
            ood_thresholds.append(class_ood_thresholds)
        
        return ood_thresholds
        
    def get_eval_results(self, ftrain,fval, ftest, food, labelstrain, labelsval,labels_test):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, fval_norm, ftest_norm, food_norm = self.normalise(ftrain, fval, ftest, food)
        class_means, class_cov,class_entropy, class_quantile_thresholds = self.get_offline_thresholds(ftrain_norm, fval_norm, labelstrain, labelsval)
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