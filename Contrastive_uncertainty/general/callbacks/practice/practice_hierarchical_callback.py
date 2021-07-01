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

class Practice_Hierarchical(pl.Callback):
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
            np.copy(labels_train_fine),
            np.copy(indices_dtest_coarse),
            np.copy(indices_dood_coarse))
        
    
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
        
        din, indices_din = self.get_conditional_scores(din,ptest_index)
        dood, indices_dood = self.get_conditional_scores(dood,pood_index)
        
        return din, dood, indices_din, indices_dood
    
    def get_conditional_scores(self,ddata, prev_indices=None):
        # import ipdb; ipdb.set_trace()
        if prev_indices is not None: # index of the previous test values
            #import ipdb; ipdb.set_trace()
            coarse_test_mapping =  self.Datamodule.coarse_mapping.numpy()
            ddata = np.stack(ddata,axis=1) # stacks the array to make a (batch,num_classes) array
            collated_ddata = []
            collated_indices = []
            # Go throuhg each datapoint hierarchically
            for i,sample_distance in enumerate(ddata):
                #import ipdb; ipdb.set_trace()
                # coarse_test_mapping==ptest_index[i]] corresponds to a boolean mask placed on sample to get only the values of interest
                conditioned_distance = sample_distance[coarse_test_mapping==prev_indices[i]] # Get the data point which have the same superclass
                # Obtain the smallest value for the conditioned distances
                min_conditioned_distance = np.min(conditioned_distance)
                sample_index = np.where(sample_distance == min_conditioned_distance)[0][0] # Obtain the index from the datapoint to get the fine class label

                collated_ddata.append(min_conditioned_distance)
                collated_indices.append(sample_index)

            ddata = np.array(collated_ddata)
            indices_ddata = np.array(collated_indices)
            #import ipdb; ipdb.set_trace()
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
        if ptest_index is not None:
            assert pood_index is not None, 'conditioning on the test data but not on OOD should not occur'
        """
            None.
        """
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, ptest_index, pood_index)
        
        return dtest, dood, indices_dtest, indices_dood


class Practice_Hierarchical_scores(Practice_Hierarchical):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):
        super().__init__(Datamodule,OOD_Datamodule,quick_callback)


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

    
        dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine))

        dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse,),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))


        dtest_conditional_fine, dood_conditional_fine, indices_dtest_conditional_fine, indices_dood_conditional_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            np.copy(indices_dtest_coarse),
            np.copy(indices_dood_coarse))

        
        #dtest_fine =  dtest_fine + np.random.randint(10, size=len(dtest_fine))
        #dood_fine = dood_fine + np.random.randint(10, size=len(dood_fine))
        #import ipdb;  ipdb.set_trace()
        ID_dict = {'ID Fine': dtest_fine, 'ID Conditional Fine': dtest_conditional_fine}
        OOD_dict = {f'{self.OOD_dataname} Fine': dood_fine, f'{self.OOD_dataname} Conditional Fine': dood_conditional_fine}
        
        #import ipdb; ipdb.set_trace()
        
        '''
        print('ID fine',ID_dict['ID Fine'])
        print('ID conditional fine',ID_dict['ID Conditional Fine'])
        print('OOD fine',OOD_dict[f'{self.OOD_dataname} Fine'])
        print('OOD Conditional Fine',OOD_dict[f'{self.OOD_dataname} Conditional Fine'])
        print('ID conditional fine',ID_dict['ID Conditional Fine'])
        #ID_dict = {'ID Fine': dtest_fine, 'ID Conditional Fine': dtest_conditional_fine}
        '''

        OOD_dict = {f'{self.OOD_dataname} Fine': dood_fine,f'{self.OOD_dataname} Conditional Fine': dood_conditional_fine}
        # https://towardsdatascience.com/merge-dictionaries-in-python-d4e9ce137374
        #all_dict = {**ID_dict,**OOD_dict} # Merged dictionary
        all_dict = {'ID Fine': dtest_fine, 'ID Conditional Fine': dtest_conditional_fine,f'{self.OOD_dataname} Fine': dood_fine,f'{self.OOD_dataname} Conditional Fine': dood_conditional_fine}

        ID_name = f'Hierarchical Fine ID {self.OOD_dataname} data scores'
        OOD_name = f'Hierarchical Fine OOD {self.OOD_dataname} data scores'
        all_name = f'Hierarchical Fine All {self.OOD_dataname} data scores'
        ID_name_counts = f'Hierarchical Fine ID {self.OOD_dataname} counts'
        OOD_name_counts = f'Hierarchical Fine OOD {self.OOD_dataname} counts'
        all_name_counts = f'Hierarchical Fine All {self.OOD_dataname} counts'
        # Replace white spaces with underscore  https://stackoverflow.com/questions/1007481/how-do-i-replace-whitespaces-with-underscore
        '''
        kde_plot(ID_dict,ID_name,ID_name.replace(" ","_"),ID_name)
        kde_plot(OOD_dict,OOD_name,OOD_name.replace(" ","_"),OOD_name)
        kde_plot(all_dict,all_name, all_name.replace(" ","_"),all_name)
        '''
        count_plot(ID_dict,ID_name_counts,ID_name_counts.replace(" ","_"),ID_name_counts)
        count_plot(OOD_dict,OOD_name_counts,OOD_name_counts.replace(" ","_"),OOD_name_counts)
        count_plot(all_dict,all_name_counts, all_name_counts.replace(" ","_"),all_name_counts)
        

def kde_plot(input_data,title_name,file_name,wandb_name):
    sns.displot(data =input_data,fill=True,common_norm=False,kind='kde', multiple="stack")
    plt.xlabel('Distance')
    plt.ylabel('Normalized Density')
    plt.xlim([0, 10])
    plt.title(f'{title_name}')
    kde_filename = f'Images/{file_name}.png'
    plt.savefig(kde_filename,bbox_inches='tight')
    #plt.show()
    plt.close()
    wandb_distance = f'{wandb_name}'
    wandb.log({wandb_distance:wandb.Image(kde_filename)})


def count_plot(input_data,title_name,file_name,wandb_name):
    sns.displot(data =input_data,fill=True,common_norm=False, multiple="stack")
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.xlim([0, 10])
    plt.title(f'{title_name}')
    filename = f'Images/{file_name}.png'
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    plt.close()
    wandb_distance = f'{wandb_name}'
    wandb.log({wandb_distance:wandb.Image(filename)})







    

