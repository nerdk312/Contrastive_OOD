from numpy.core.numeric import indices
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
import random 

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
import eif as iso

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_fpr, get_pr_sklearn, get_roc_plot, get_roc_sklearn, table_saving, calculate_class_ROC



class IForest(pl.Callback):
    def __init__(self,Datamodule, OOD_Datamodule, quick_callback:bool=True,
        vector_level = 'fine', label_level= 'fine'):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
   
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name

        self.vector_level = vector_level
        self.label_level = label_level

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
        features_train, labels_train = self.get_features(pl_module, train_loader, self.vector_level)
        features_test, labels_test = self.get_features(pl_module, test_loader, self.vector_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level)

        non_class_dtest, non_class_dood, dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train)
        )
        self.AUROC_saving(non_class_dtest, dtest, indices_dtest, labels_test, non_class_dood, dood, indices_dood,
        f'Isolation Forest {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC', 
        f'Isolation Forest {self.vector_level} {self.label_level} OOD {self.OOD_dataname} AUROC Table')

    
    
    def AUROC_saving(self,non_class_ID_scores, ID_scores, indices_ID, labels,
        non_class_OOD_scores, OOD_scores,indices_OOD, wandb_name, table_name):
        '''
        args:
            non_subclustered_ID_Scores : Non clustered score for a particular class
            ID scores : scores for the subclusters of a particular class
            indices ID: indices for the subcluster assignment for the ID dataset
            labels : labels for the different number of subclusters
        '''

        table_data = {'Class':[], 'AUROC':[], 'ID Samples Fraction':[],'OOD Samples Fraction':[]}
        
        # Scores for the subclusters for the ID and OOD data
        din_class = [ID_scores[indices_ID==i] for i in np.unique(labels)]
        dood_class = [OOD_scores[indices_OOD ==i] for i in np.unique(labels)] 
        
        # Calculate AUROC score for each individual class
        for class_num in range(len(din_class)):
            class_AUROC = calculate_class_ROC(din_class[class_num],dood_class[class_num])
            
            class_ID_fraction = len(din_class[class_num])/len(ID_scores)
            
            #import ipdb; ipdb.set_trace()
            class_OOD_fraction = len(dood_class[class_num])/len(OOD_scores)
            
            table_data['Class'].append(f'Class {class_num}')
            table_data['AUROC'].append(round(class_AUROC,2))
            table_data['ID Samples Fraction'].append(round(class_ID_fraction,2))
            table_data['OOD Samples Fraction'].append(round(class_OOD_fraction,2))
        
        # Obtain the AUROC for all the classes together rather than individually
        all_class_AUROC = get_roc_sklearn(ID_scores,OOD_scores)
        table_data['Class'].append('All Classes')
        table_data['AUROC'].append(round(all_class_AUROC,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)

        # Obtain the AUROC score for the case where the isolation forest is trained on data from all classes
        #import ipdb; ipdb.set_trace()
        non_class_AUROC = get_roc_sklearn(non_class_ID_scores, non_class_OOD_scores)
        table_data['Class'].append('All No Classes')
        table_data['AUROC'].append(round(non_class_AUROC ,2))
        table_data['ID Samples Fraction'].append(1.0)
        table_data['OOD Samples Fraction'].append(1.0)

        # Table saving
        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)



        '''
        clf = IsolationForest(contamination=0.0).fit(features_train)
        dtest = clf.predict(features_test)
        dood = clf.predict(features_ood)


        AUROC = get_roc_sklearn(dtest,dood)
        import ipdb; ipdb.set_trace()
        '''
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
        return ftrain, ftest, food

    def get_eval_results(self,ftrain, ftest, food, labelstrain): 
        # Normalise the data       
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        '''
        non_class_clf = IsolationForest().fit(ftrain_norm)
        non_class_dtest = non_class_clf.predict(ftest_norm)
        non_class_dood = non_class_clf.predict(food_norm)
        '''
        #import ipdb; ipdb.set_trace()
        ftrain_norm = np.array(ftrain_norm, dtype='float64')
        ftest_norm = np.array(ftest_norm, dtype='float64')
        food_norm = np.array(food_norm, dtype='float64')
        F = iso.iForest(ftrain_norm,ntrees=200, sample_size=256, ExtensionLevel=0) # Train the isolation forest on indomain dat
        non_class_dtest = F.compute_paths(X_in=ftest_norm) # Nawid - compute the paths for the nominal datapoints
        non_class_dood = F.compute_paths(X_in=food_norm)
        #non_class_AUROC = get_roc_sklearn(non_class_dtest,non_class_dood)
        #print('non classs AUROC', non_class_AUROC)
        #print(' non class dtest', non_class_dtest)
        #print('non class dood', non_class_dood)

        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm,labelstrain)
        
        return non_class_dtest,non_class_dood, dtest, dood, indices_dtest, indices_dood

    
    # Calculate per class isolation distance
    def get_scores(self, ftrain, ftest, food, ypred): # Add additional variables for the parent index
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        # Obtain the mahalanobis distance scores for the different classes on the train data

        # Fit several classifiers for the different classes
        #clfs = [IsolationForest().fit(x) for x in xc]
        clfs = [iso.iForest(x,ntrees=200, sample_size=256, ExtensionLevel=0) for x in xc]
        # Make predictions using each of the class specific classifiers

        din = [clf.compute_paths(X_in = ftest) for clf in clfs]
        dood = [clf.compute_paths(X_in = food) for clf in clfs]
        #din = [clf.predict(ftest) for clf in clfs]
        #dood = [clf.predict(food) for clf in clfs]

        # Find out which class the data is present
        indices_din = np.argmax(din,axis=0)
        indices_dood = np.argmax(dood,axis=0)
        # Obtain the max score based on how well the data is divided up
        din = np.max(din,axis=0)
        dood = np.max(dood,axis=0)
        return din, dood, indices_din, indices_dood
    
    


        



