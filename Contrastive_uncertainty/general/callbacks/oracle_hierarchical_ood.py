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


# Calculates the improvement in the classification accuracy in a hierarchical manner
class Oracle_Hierarchical(pl.Callback):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
   
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.OOD_dataname = self.OOD_Datamodule.name


    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train_coarse, labels_train_coarse = self.get_features(pl_module, train_loader,'coarse')
        features_train_fine, labels_train_fine = self.get_features(pl_module, train_loader,'fine')

        features_test_coarse, labels_test_coarse = self.get_features(pl_module, test_loader,'coarse')
        features_test_fine, labels_test_fine = self.get_features(pl_module, test_loader,'fine')

        features_ood_coarse, labels_ood_coarse = self.get_features(pl_module, ood_loader, 'coarse')
        features_ood_fine, labels_ood_fine = self.get_features(pl_module, ood_loader, 'fine')
        
        dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))

        dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine))


        dtest_conditional_fine, _, indices_dtest_conditional_fine, _ = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            # Additional used for conditioning on the true coarse labels to see if it improves the results 
            np.copy(labels_test_coarse))

        self.oracle_conditional_accuracy_difference(indices_dtest_fine,indices_dtest_conditional_fine,labels_test_fine)

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        return mahalanobis_test_accuracy
    
     # Calculates conditional accuracy for the data
    def oracle_conditional_accuracy_difference(self, unconditional_pred, conditional_pred, labels):
        fine_unconditional_accuracy = self.mahalanobis_classification(unconditional_pred, labels)
        fine_conditional_accuracy = self.mahalanobis_classification(conditional_pred,labels)
        conditional_diff = fine_conditional_accuracy - fine_unconditional_accuracy
        
        wandb.run.summary['Fine Unconditional Accuracy'] = fine_unconditional_accuracy
        wandb.run.summary['Oracle Fine Conditional Accuracy'] = fine_conditional_accuracy
        wandb.run.summary['Oracle Fine Conditional Improvement'] = conditional_diff


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
    

    def get_scores(self, ftrain, ftest, food, ypred, ptest_index = None, pood_index=None): # Add additional variables for the parent index
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        # Obtain the mahalanobis distance scores for the different classes on the train data
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
        
        
        # Obtain the mahalanobis distance scores for the different classes on the test data
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
        #import ipdb; ipdb.set_trace()
        din, indices_din = self.get_conditional_scores(din,ptest_index)
        dood, indices_dood = self.get_conditional_scores(dood,pood_index)
        
        return din, dood, indices_din, indices_dood
        
    
    def get_conditional_scores(self,ddata, prev_indices=None):
        # import ipdb; ipdb.set_trace()
        if prev_indices is not None: # index of the previous test values
            coarse_test_mapping =  self.Datamodule.coarse_mapping.numpy()
            ddata = np.stack(ddata,axis=1) # stacks the array to make a (batch,num_classes) array
            collated_ddata = []
            collated_indices = []
            # Go throuhg each datapoint hierarchically
            for i,sample_distance in enumerate(ddata):
                # coarse_test_mapping==ptest_index[i]] corresponds to a boolean mask placed on sample to get only the values of interest
                conditioned_distance = sample_distance[coarse_test_mapping==prev_indices[i]] # Get the data point which have the same superclass
                # Obtain the smallest value for the conditioned distances
                min_conditioned_distance = np.min(conditioned_distance)
                sample_index = np.where(sample_distance == min_conditioned_distance)[0][0] # Obtain the index from the datapoint to get the fine class label

                collated_ddata.append(min_conditioned_distance)
                collated_indices.append(sample_index)

            ddata = np.array(collated_ddata)
            indices_ddata = np.array(collated_indices)
        else:    
            indices_ddata = np.argmin(ddata,axis = 0)  
            ddata = np.min(ddata, axis=0) # Nawid - calculate the minimum distance 

        return ddata, indices_ddata

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
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain,ptest_index = None, pood_index=None):
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, ptest_index, pood_index)

        return dtest, dood, indices_dtest, indices_dood


# Calculates the class wise AUROC when using the oracle as well as the improvement, also calculates the classification accuracy
class Oracle_Hierarchical_Metrics(Oracle_Hierarchical):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,quick_callback)


    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train_fine, labels_train_fine = self.get_features(pl_module, train_loader,'fine')

        # Coarse test labels required for conditioning
        features_test_coarse, labels_test_coarse = self.get_features(pl_module, test_loader,'coarse')
        features_test_fine, labels_test_fine = self.get_features(pl_module, test_loader,'fine')

        features_ood_fine, labels_ood_fine = self.get_features(pl_module, ood_loader, 'fine')
        
        
        dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine))

        # Condition on the true test labels
        dtest_conditional_fine, _, indices_dtest_conditional_fine, _ = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            # Additional used for conditioning on the true coarse labels to see if it improves the results 
            np.copy(labels_test_coarse))
        
        self.data_saving(dtest_fine,indices_dtest_fine,
            dtest_conditional_fine,indices_dtest_conditional_fine,
            dood_fine,indices_dood_fine,labels_test_fine,
            f'Hierarchical Oracle Metrics OOD {self.OOD_dataname}',f'Hierarchical Oracle Metrics OOD {self.OOD_dataname} Table')

        #auroc, aupr = get_roc_sklearn(dtest, dood)
    
    def data_saving(self,ID_scores,indices_ID, 
        ID_conditional_scores,indices_conditional_ID,
        OOD_scores,indices_OOD,labels, wandb_name,table_name):
        
        # Metrics should be all the different class AUROC, the general AUROC as well as the classification accuracy
        table_data = {'Metric':[], 'Oracle': [], 'Oracle Improvement':[]}
        
        # class scores for the ID, conditional ID and OOD data
        din_class = [ID_scores[indices_ID==i] for i in np.unique(labels)]
        din_conditional_class = [ID_conditional_scores[indices_conditional_ID==i] for i in np.unique(labels)]
        dood_class = [OOD_scores[indices_OOD ==i] for i in np.unique(labels)]

        # Class AUROC values
        for class_num in range(len(din_class)):
            # Calculate AUROC for the normal class case as well as the Oracle class case
            class_AUROC = self.calculate_class_ROC(din_class[class_num],dood_class[class_num])
            oracle_class_AUROC = self.calculate_class_ROC(din_conditional_class[class_num],dood_class[class_num])
            #import ipdb; ipdb.set_trace()
            # Calculate the difference in the values
            oracle_class_improvement = oracle_class_AUROC - class_AUROC
            table_data['Metric'].append(f'Class {class_num} AUROC')
            table_data['Oracle'].append(oracle_class_AUROC)
            table_data['Oracle Improvement'].append(oracle_class_improvement)
        
        # Non class specific metrics
        table_data['Metric'].append('AUROC')
        # Calculate general scores
        auroc = get_roc_sklearn(ID_scores,OOD_scores)
        oracle_auroc = get_roc_sklearn(ID_conditional_scores,OOD_scores)
        oracle_auroc_improvement = oracle_auroc - auroc

        table_data['Oracle'] = oracle_auroc
        table_data['Oracle Improvement'] = oracle_auroc_improvement

        table_data['Metric'].append('Classification Accuracy')
        classification = self.mahalanobis_classification(indices_ID,labels)
        oracle_classification = self.mahalanobis_classification(indices_conditional_ID,labels)
        oracle_classification_improvement = oracle_auroc - auroc
        table_data['Oracle'] = oracle_classification
        table_data['Oracle Improvement'] = oracle_classification_improvement

        # Table saving
        table_df = pd.DataFrame(table_data)
        import ipdb; ipdb.set_trace()
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)
        #print('HIERARCHICAL METRIC BEING USED')
        

    # Function to compute classwise AUROC
    def calculate_class_ROC(self, class_ID_scores, class_OOD_scores):
        if len(class_ID_scores) ==0 or len(class_OOD_scores)==0:
            class_AUROC = -1.0
        else:
            class_AUROC = get_roc_sklearn(class_ID_scores, class_OOD_scores)
        return class_AUROC







'''
class Oracle_Hierarchical_OOD(Oracle_Hierarchical):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,quick_callback)


    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # Obtain representations of the data
        features_train_coarse, labels_train_coarse = self.get_features(pl_module, train_loader,'coarse')
        features_train_fine, labels_train_fine = self.get_features(pl_module, train_loader,'fine')

        features_test_coarse, labels_test_coarse = self.get_features(pl_module, test_loader,'coarse')
        features_test_fine, labels_test_fine = self.get_features(pl_module, test_loader,'fine')

        features_ood_coarse, labels_ood_coarse = self.get_features(pl_module, ood_loader, 'coarse')
        features_ood_fine, labels_ood_fine = self.get_features(pl_module, ood_loader, 'fine')
        
        dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))

        dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine))


        dtest_conditional_fine, _, indices_dtest_conditional_fine, _ = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            # Additional used for conditioning on the true coarse labels to see if it improves the results 
            np.copy(labels_test_coarse))
'''       