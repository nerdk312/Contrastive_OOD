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
import scipy

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

        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level)
        self.get_eval_results(features_train, labels_train)

    
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
        #print('variance 1',class_variance_approach1)

        class_variance_approach2 = [np.var(ftrain[ypred==i]) for i in np.unique(ypred)]
        #print('variance 2',class_variance_approach2)
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
    

    def data_saving(self,class_variance_scores1, class_variance_scores2, wandb_name,table_name):
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
            score1,score2 =  class_variance_scores1[class_num], class_variance_scores2[class_num]
            table_data['Variance Approach 1'].append(score1)
            table_data['Variance Approach 2'].append(score2)        

        table_data
        # Table saving
        table_df = pd.DataFrame(table_data)
        table_df = table_df.round(3)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)
    
    def get_eval_results(self,ftrain,labelstrain):
        ftrain_norm = self.normalise(ftrain)
        class_var1, class_var2 = self.get_variance(ftrain_norm,labelstrain)
        #class_var = [np.var(ftrain_norm[labelstrain==i]) for i in np.unique(labelstrain)]
        self.data_saving(class_var1,class_var2,f'Class Variance {self.vector_level} {self.label_level}',f'Class Variance {self.vector_level} {self.label_level} Table')



class Dataset_class_radii(Dataset_class_variance):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True, vector_level='fine',label_level='fine', lower_percentile:int = 75, upper_percentile:int = 95):
    
        super().__init__(Datamodule, OOD_Datamodule,
        quick_callback, vector_level,label_level)

        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile

    def forward_callback(self, trainer, pl_module):

        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        
        train_loader = self.Datamodule.deterministic_train_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level)
        self.get_eval_results(features_train, labels_train)

    def get_radii(self, ftrain, ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        class_centroids = [np.mean(x,axis=0,keepdims=True) for x in xc]
        diff = [xc[class_num] - class_centroids[class_num] for class_num in range(len(class_centroids))]
        radii = [np.linalg.norm(class_data,axis=1) for class_data in diff] # Changes shape from (class_batch, dim) to (class_batch)
        return radii

    def get_eval_results(self, ftrain, labelstrain):
        ftrain_norm = self.normalise(ftrain)
        radii = self.get_radii(ftrain_norm, labelstrain)

        
        lower_radii = [round(np.percentile(class_radii,self.lower_percentile),2) for class_radii in radii]
        upper_radii = [round(np.percentile(class_radii,self.upper_percentile),2) for class_radii in radii]
        
        self.data_saving(lower_radii,upper_radii,f'ID Training Class Radii {self.vector_level} {self.vector_level}', f'ID Training Class Radii {self.vector_level} {self.vector_level} Table')

    def data_saving(self,lower_radii_values,upper_radii_values, wandb_name, table_name):
        '''
        args:
            radii_percentile : List which contains the radii of the self.percentile radius of the data
            wandb name: name of data table
            table_name: name of the saved table
        '''
        table_data = {'Class':[], f'{self.lower_percentile}th Percentile Radii':[],f'{self.upper_percentile}th Percentile Radii':[]}
         # Save variance score for the
        for class_num in range(len(lower_radii_values)):    
            table_data['Class'].append(f'Class {class_num}')

            table_data[f'{self.lower_percentile}th Percentile Radii'].append(lower_radii_values[class_num])
            table_data[f'{self.upper_percentile}th Percentile Radii'].append(upper_radii_values[class_num])        

        # Table saving
        #import ipdb; ipdb.set_trace()
        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name:table})
        table_saving(table_df,table_name)


class Centroid_distances(Dataset_class_variance):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True, vector_level='fine',label_level='fine'):
        
        super().__init__(Datamodule, OOD_Datamodule,
            quick_callback, vector_level,label_level)
    
    def forward_callback(self, trainer, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()


        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level)
        self.get_eval_results(features_train, labels_train)


    def get_cluster_centroids(self,ftrain,ypred):
        centroids = []
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        centroids = np.array([np.mean(x, axis=0) for x in xc])
        return centroids


    def centroid_distances(self,ftrain,ypred):
        # Makes barchart of deviations from the average vector of the training set
        avg_vector = np.mean(ftrain, axis=0)
        avg_vector = np.reshape(avg_vector, (1, -1))
        centroids = self.get_cluster_centroids(ftrain,ypred)
        #import ipdb; ipdb.set_trace()
        #diff = np.abs(centroids - avg_vector) # Calculates the absolute difference element wise to ensure that the mean does not cancel out
        #total_diff = np.mean(diff, axis=1)

        
        diff = centroids - avg_vector
        centroid_dist = np.linalg.norm(diff,axis=1) # shape (num_classes,)
        
        labels = [i for i in np.unique(ypred)] 
        #data =[[label, val] for (label ,val) in zip(labels,total_diff)] # iterates through the different labels as well as the different values for the labels
        data = [[label, val] for (label ,val) in zip(labels,centroid_dist)] # iterates through the different labels as well as the different values for the labels
        table = wandb.Table(data=data, columns = ["Label", "Distance"])
        wandb.log({"Centroid Distances Average vector" : wandb.plot.bar(table, "Label", "Distance",
                               title="Centroid Distances Average vector")})

        
    def get_eval_results(self,ftrain,labelstrain):
        ftrain_norm = self.normalise(ftrain)
        self.centroid_distances(ftrain_norm,labelstrain)


class Class_Radii_histograms(Dataset_class_variance):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True, vector_level='fine',label_level='fine'):
    
        super().__init__(Datamodule, OOD_Datamodule,
        quick_callback, vector_level,label_level)

    def forward_callback(self, trainer, pl_module):

        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}}  
        
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader, self.vector_level)

        in_class_radii_scores, ood_class_radii_scores = self.get_eval_results(features_train,features_test, features_ood, labels_train)
        self.data_saving(in_class_radii_scores, ood_class_radii_scores,labels_train,f'Class Radii Scores {self.vector_level} {self.label_level} {self.OOD_dataname}')


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
    
    def get_centroids(self,ftrain, ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        class_centroids = [np.mean(x,axis=0,keepdims=True) for x in xc]
        return class_centroids # List of class centroids

    def class_radii(self, centroids, fdata, indices_data, labels):
        xc = [fdata[indices_data == class_num] for class_num in np.unique(labels)] # Get all the data for the specific labesl
        class_radii = [np.around(np.linalg.norm(centroids[class_num] - xc[class_num],axis=1),decimals=3) for class_num in np.unique(labels)] # if len(xc[class_num])> 0 else np.nan]
        #import ipdb; ipdb.set_trace()
        for class_num in range(len(class_radii)):
            if len(class_radii[class_num]) == 0: 
                class_radii[class_num] = np.array([-1.1]) 

        return class_radii
        
    def get_eval_results(self, ftrain, ftest, food, labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain, ftest, food)
        dtest, dood, indices_dtest, indices_dood= self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        
        class_centroids = self.get_centroids(ftrain_norm, labelstrain)
        test_class_radii_scores = self.class_radii(class_centroids, ftest_norm, indices_dtest, labelstrain)
        ood_class_radii_scores = self.class_radii(class_centroids, food_norm, indices_dood, labelstrain)
        return test_class_radii_scores, ood_class_radii_scores
        
    
    def data_saving(self, in_class_radii_scores, ood_class_radii_scores, labels, wandb_dataname):
        # obtain the data score for the subclusters
        in_radii  = [pd.DataFrame(in_class_radii_scores[class_num]) for class_num in np.unique(labels)] # Nawid - training data which have been predicted to belong to a particular class
        ood_radii  = [pd.DataFrame(ood_class_radii_scores[class_num]) for class_num in np.unique(labels)]
        collated_radii = [*in_radii, *ood_radii] #[in_radii] + [ood_radii]
        # Concatenate all the dataframes (which places nans in situations where the columns have different lengths)
        radii_df = pd.concat(collated_radii,axis =1 )
        #https://stackoverflow.com/questions/30647247/replace-nan-in-a-dataframe-with-random-values
        radii_df = radii_df.applymap(lambda l: l if not np.isnan(l) else random.uniform(-2,-1))
        
        
        in_columns = [f'Class {i} ID Radii scores' for i in np.unique(labels)]
        ood_columns = [f'Class {i} OOD Radii scores' for i in np.unique(labels)]
        radii_df.columns = [*in_columns, *ood_columns]
        #columns1 = [*in_columns, *ood_columns]
        #columns2 = [in_columns] + [ood_columns]
        radii_table = wandb.Table(data=radii_df)
        wandb.log({wandb_dataname:radii_table})







# Alternative approach of calculating the centroid distance, where the distance of a centroid is based on the distance to other centroids of the data
class Centroid_relative_distances(Centroid_distances):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True, vector_level='fine',label_level='fine'):
        
        super().__init__(Datamodule, OOD_Datamodule,
            quick_callback, vector_level,label_level)

    def centroid_distances(self,ftrain,ypred):
        # Makes barchart of deviations from the average vector of the training set
        centroids = self.get_cluster_centroids(ftrain,ypred) # shape (num classes, embeding size)
        dist_matrix = scipy.spatial.distance.cdist(centroids, centroids) # shape (num_class,num_class) where for each i and j, the the metric dist(u=XA[i], v=XB[j]) is computed and stored in the  ij th entry, computes distance between centroid i and centroid j
        average_centroid_distances = np.mean(dist_matrix,axis=1) # calculates the average of the distances of each centroid to all the other centroids
        return average_centroid_distances

        

    def data_saving(self,average_centroids_distaces,ypred):
        labels = [i for i in np.unique(ypred)] 
        #data =[[label, val] for (label ,val) in zip(labels,total_diff)] # iterates through the different labels as well as the different values for the labels
        data = [[label, val] for (label ,val) in zip(labels,average_centroids_distaces)] # iterates through the different labels as well as the different values for the labels
        table = wandb.Table(data=data, columns = ["Label", "Distance"])
        wandb.log({"Class Centroid Relative Distances" : wandb.plot.bar(table, "Label", "Distance",
                               title="Class Centroid Relative Distances")})
    
        
    def get_eval_results(self,ftrain,labelstrain):
        ftrain_norm = self.normalise(ftrain)
        average_centroid_distances = self.centroid_distances(ftrain_norm,labelstrain)
        self.data_saving(average_centroid_distances,labelstrain)