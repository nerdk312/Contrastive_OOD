from numpy.lib.financial import _ipmt_dispatcher
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import subprocess
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
import scipy

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import glob

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, get_fpr, get_pr_sklearn, get_roc_plot, get_roc_sklearn, table_saving
from Contrastive_uncertainty.general.callbacks.hierarchical_ood import kde_plot, count_plot
from Contrastive_uncertainty.general.callbacks.one_dim_mahalanobis_callback import One_Dim_Mahalanobis
from Contrastive_uncertainty.general.callbacks.compare_distributions import ks_statistic, ks_statistic_kde, js_metric, kl_divergence


# Calculating the similarity between the different classes using the one dimensional mahalanobis distance
class One_Dim_Mahalanobis_Similarity(pl.Callback):
    def __init__(self, Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):
    
        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        # Chooses what vector representation to use as well as which level of label hierarchy to use
        self.vector_level = vector_level
        self.label_level = label_level
    
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

        # Obtain representations of the data
        features_train, labels_train = self.get_features(pl_module, train_loader,self.vector_level, self.label_level)
        features_test, labels_test = self.get_features(pl_module, test_loader,self.vector_level, self.label_level)
        
        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(labels_train),
            np.copy(labels_test))
    

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


    def get_scores(self, ftrain, ftest, labelstrain, labelstest):
        # Nawid - get all the features which belong to each of the different classes
        xctrain = [ftrain[labelstrain == i] for i in np.unique(labelstrain)] # Nawid - training data which have been predicted to belong to a particular class
        xctest = [ftest[labelstest == i] for i in np.unique(labelstest)]
        cov = [np.cov(x.T, bias=True) for x in xctrain] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xctrain] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        
        # Obtain the eigenvalues and eigenvectors for all the different classes of the data
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of

            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals, axis=1))
            eigvectors.append(class_eigvectors)
        
        # Obtain the scores for the different dimensions related to each particular label for the train and test data
        dtrain = [np.abs(np.matmul(eigvectors[class_num].T,(xctrain[class_num] - means[class_num]).T)**2/eigvalues[class_num]).T for class_num in range(len(cov))]  # Add an additional transpose to make it have a shape of (class_batch, dimension)
        dtest = [np.abs(np.matmul(eigvectors[class_num].T,(xctest[class_num] - means[class_num]).T)**2/eigvalues[class_num]).T for class_num in range(len(cov))]  # 
        
        return dtrain, dtest
        
        '''
        # Means of different dimensions as well as the scores for the case of the approach
        one_dim_means = [np.mean(dtrain[class_num],axis=0,keepdims=True) for class_num in range(len(dtrain))] # Means of all the dimensions
        one_dim_var = [np.var(dtrain[class_num],axis=0, keepdims=True) for class_num in range(len(dtrain))] # 
        '''
    
    def normalise(self,ftrain,ftest):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        
        return ftrain, ftest

    def generate_video(self,dtrain, dtest,labelstrain,labelstest):
        # Contains all the KL values
        collated_KL = []
        
        dims = dtrain[0].shape[-1]
        #import ipdb; ipdb.set_trace()
        for class_i in np.unique(labelstrain):
            class_KL = []
            for class_j in np.unique(labelstest):
                # Calculate the KL divergence for each dimension between train and test dataset of the particular classes
                KL = [kl_divergence(dtrain[class_i][:,dim],dtest[class_j][:,dim]) for dim in range(dims)] #range(len(class_eigvals))]
                class_KL.append(KL)
                            
                
            collated_KL.append(class_KL)
        
        data = np.array(collated_KL) # shape (num class i, num class j, dim) also equivalent (Train class, test classes, dim)
        data = np.around(data,decimals=1)
        column_names = [f'{i}' for i in range(len(np.unique(labelstest)))]
        index_names = [f'{i}' for i in range(len(np.unique(labelstrain)))]
        for i in range(dims):
            table_df = pd.DataFrame(data[:,:,i], index = index_names, columns=column_names)
            # Show the data in the case where the data is not being plotted
            sns.heatmap(table_df,annot=False,fmt=".1f")
            plt.xlabel('Test')
            plt.ylabel('Train')
            plt.title(f'1D Mahalanobis Dimension {i} Divergence')
            #filename = f'Images/1d_mahalanobis_dim{i}_divergence.png'
            #filename = f'Images/file{i}.png'
            filename = 'Images' + "/file%02d.png" % i

            # Cannot use bbox inches as this prevents the video from being made
            plt.savefig(filename)#,bbox_inches='tight')
            plt.close()
        
        #os.chdir("Images")
        video_filename = 'Images/1D_train_test_divergence.mp4' 
        subprocess.call([
            'ffmpeg', '-framerate', '4', '-i', 'Images/file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p', '-y',
            video_filename
        ])  # Added the -y argument to lead to overwriting file if present
        # The third parameter in call controls the framerate
        wandb_filename = '1D Mahalanobis Divergence Heatmaps'
        wandb.log({wandb_filename: wandb.Video(video_filename, fps=4, format="mp4")})
        
        for file_name in glob.glob("Images/*.png"):
            os.remove(file_name)
        '''
        # Removes the mp4 file after to prevent issue of the different datamodules leading to it being used several times
        for filename in glob.glob('Images/*.mp4'):
            os.remove(filename)
        '''

    def get_eval_results(self, ftrain, ftest, labelstrain,labelstest):
        ftrain_norm, ftest_norm= self.normalise(ftrain, ftest)
        # Nawid - obtain the scores for the test data and the OOD data
        dtrain, dtest = self.get_scores(ftrain_norm, ftest_norm, labelstrain, labelstest)    
        self.generate_video(dtrain,dtest,labelstrain, labelstest)


# Examines the class wise similarity between the classes for different datasets
class Class_One_Dim_Mahalanobis_OOD_Similarity(One_Dim_Mahalanobis):
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

        self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_test))
    
    # Use the mahalanobis distance to get predictions for the test and the train datasets
    def get_predictions(self,ftrain, ftest, food, ypred):
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

        return indices_din, indices_dood


    def get_scores(self, ftrain, ftest,food, labelstrain, labelstest,ood_pred):
        # Nawid - get all the features which belong to each of the different classes
        xctrain = [ftrain[labelstrain == i] for i in np.unique(labelstrain)] # Nawid - training data which have been predicted to belong to a particular class
        xctest = [ftest[labelstest == i] for i in np.unique(labelstest)]
        xcood = [food[ood_pred == i] for i in np.unique(labelstest)] # get the predictions for the particular case for the different labels


        cov = [np.cov(x.T, bias=True) for x in xctrain] # Cov and means part should be fine
        means = [np.mean(x,axis=0,keepdims=True) for x in xctrain] # Calculates mean from (B,embdim) to (1,embdim)
        
        eigvalues = []
        eigvectors = []
        
        # Obtain the eigenvalues and eigenvectors for all the different classes of the data
        for class_cov in cov:
            class_eigvals, class_eigvectors = np.linalg.eigh(class_cov) # Each column is a normalised eigenvector of

            # Reverse order as the eigenvalues and eigenvectors are in ascending order (lowest value first), therefore it would be beneficial to get them in descending order
            #class_eigvals, class_eigvectors = np.flip(class_eigvals, axis=0), np.flip(class_eigvectors,axis=0)
            eigvalues.append(np.expand_dims(class_eigvals, axis=1))
            eigvectors.append(class_eigvectors)
        
        # Obtain the scores for the different dimensions related to each particular label for the train and test data, as well as prediction from the model
        dtrain = [np.abs(np.matmul(eigvectors[class_num].T,(xctrain[class_num] - means[class_num]).T)**2/eigvalues[class_num]).T for class_num in range(len(cov))]  # Add an additional transpose to make it have a shape of (class_batch, dimension)
        dtest = [np.abs(np.matmul(eigvectors[class_num].T,(xctest[class_num] - means[class_num]).T)**2/eigvalues[class_num]).T for class_num in range(len(cov))]  # 
        dood = [np.abs(np.matmul(eigvectors[class_num].T,(xcood[class_num] - means[class_num]).T)**2/eigvalues[class_num]).T for class_num in range(len(cov))]
        

        return dtrain, dtest, dood
    

    def get_eval_results(self, ftrain, ftest,food,labelstrain,labelstest):
        ftrain_norm, ftest_norm, food_norm= self.normalise(ftrain, ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        indices_dtest, indices_dood = self.get_predictions(ftrain_norm,ftest_norm,food_norm,labelstrain)
        
        dtrain, dtest,dood = self.get_scores(ftrain_norm, ftest_norm,food_norm, labelstrain, labelstest, indices_dood)    
        
        self.class_similarity(dtrain,dtest,dood)
    
    def class_similarity(self, dtrain, dtest, dood):
        #table_data = {'Dimension':[], 'Class 0':[],'Class 1':[], 'Class 2':[],'Class 3':[]}


        dims = dtrain[0].shape[-1]
        test_KL = []
        ood_KL = []
        for class_i in range(len(dtrain)):
            class_test_KL = [kl_divergence(dtrain[class_i][:,dim],dtest[class_i][:,dim]) for dim in range(dims)]
            
            if len(dood[class_i]) > 0:
                class_ood_KL = [kl_divergence(dtrain[class_i][:,dim],dood[class_i][:,dim]) for dim in range(dims)]
            else:
                class_ood_KL = [-1.0] * dims
            
            #table_data
            test_KL.append(class_test_KL)
            ood_KL.append(class_ood_KL)
            #import ipdb; ipdb.set_trace()
        
        '''
        test_KL = np.array(test_KL) # shape (num classes, num dimensions)
        table_df = pd.DataFrame(test_KL.T) # Shape (num dimensions, num classes)
        table_df = table_df.round(3)
        table = wandb.Table(dataframe=table_df)
        wandb.log({f'practice NEW table {self.OOD_dataname} ':table})
        '''
        # import ipdb; ipdb.set_trace()


        
        xs = np.arange(len(test_KL[0]))
        ys = [test_KL[0],ood_KL[1]]
        # Plots multiple lines for the mahalanobis distance of the data # https://wandb.ai/wandb/plots/reports/Custom-Multi-Line-Plots--VmlldzozOTMwMjU
        wandb.log({f"1D Mahalanobis Similarity {self.OOD_dataname}" : wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys= ["ID data Mahalanobis per dim", f"{self.OOD_dataname} OOD data Mahalanobis per dim"],
                        title= f"1D Mahalanobis Similarity - {self.OOD_dataname} OOD data",
                        xname= "Dimension")})
        
        '''
        test_KL, ood_KL = np.array(test_KL), np.array(ood_KL)
        test_KL, ood_KL = np.around(test_KL, decimals=1), np.around(ood_KL,decimals=1)
        import ipdb; ipdb.set_trace()
        '''

        



        

        

