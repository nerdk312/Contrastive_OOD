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

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k


class Typicality_OVR(pl.Callback):
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
    
    '''
    def on_validation_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 
    '''

    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self, trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 
        
        train_loader = self.Datamodule.train_dataloader()
        val_loader = self.Datamodule.val_dataloader()
        # Use the test transform validataion loader
        if isinstance(val_loader, tuple) or isinstance(val_loader, list):
                    _, val_loader = val_loader
        
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
            
        #fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood =
        #return fpr95,auroc,aupr

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
  
    def get_offline_thresholds(self, ftrain, fval, ypred, yval):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Calculate the means of the data
        means = [np.mean(x,axis=0,keepdims=True) for x in xc] # Calculates mean from (B,embdim) to (1,embdim)

        cov = [np.cov(x.T, bias=True) for x in xc]
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function formula for entropy of a gaussian distribution where |sigma| represents determinant of sigma
        #entropy = [0.5*np.log(np.linalg.det(sigma)) for sigma in cov]
        entropy = []

        for class_num in range(len(np.unique(ypred))):
            xc_class = xc[class_num]
            dtrain = np.sum(
                (xc_class - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (xc_class - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
            # Calculates approximation for the entropy
            class_entropy = -np.mean(0.5*(dtrain**2))
            entropy.append(class_entropy)
            
        # Separate into different classes
        
        thresholds = [] # List of all threshold values
        
        xval_c = [fval[yval == i] for i in np.unique(yval)]
        # Iterate through the classes
        for class_num in range(len(np.unique(yval))):
            # get the number of indices for the class
            xval_class = xval_c[class_num] 
            indices = np.arange(len(xval_c[class_num]))
            class_thresholds = [] #  List of class threshold values
            for k in range(self.bootstrap_num):
                vals = np.random.choice(indices, size=self.typicality_bsz, replace=True)
                bootstrap_data = xval_class[vals]
                #import ipdb; ipdb.set_trace()
                # Calculates the mahalanobis distance

                dval = np.sum(
                (bootstrap_data - means[class_num]) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(cov[class_num]).dot(
                        (bootstrap_data - means[class_num]).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot product by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1)
                
                # calculate negative log likelihood
                nll = - np.mean(0.5*(dval**2))
                threshold_k = np.abs(nll- entropy[class_num])
                #threshold_k = np.abs(np.mean(dval)- entropy[class_num])
                #threshold_k = np.abs(loglikelihood- entropy[class_num])
                class_thresholds.append(threshold_k)
            
            thresholds.append(class_thresholds)
        
        # Calculating the CDF
        final_threshold = [np.quantile(np.array(class_threshold_values),0.99) for class_threshold_values in thresholds]
        return means, cov, entropy, final_threshold


    def get_online_test_thresholds(self, means, cov, entropy, ftest, ytest):
        test_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features
        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]
        # Iterate through the classes
        for class_num in range(len(np.unique(ytest))):
            # Get data for a particular class
            xtest_class = xtest_c[class_num] 
            class_test_thresholds = self.get_class_thresholds(xtest_class, means[class_num], cov[class_num],entropy[class_num]) # Get class thresholds for the particular class
            test_thresholds.append(class_test_thresholds)

        return test_thresholds
    
    def get_online_test_ood_thresholds(self, means, cov, entropy, ftest, ytest):
        test_ood_thresholds = []
        # All the data for the different classes of the test features
        #import ipdb; ipdb.set_trace()
        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]
        # Iterate through the classes
        for class_num in range(len(np.unique(ytest))):
            # Remove a subarray related to a particular class (Based on https://stackoverflow.com/questions/11903083/find-the-set-difference-between-two-large-arrays-matrices-in-python)
            a1_rows = ftest.view([('', ftest.dtype)] * ftest.shape[1])
            a2_rows = xtest_c[class_num].view([('', xtest_c[class_num].dtype)] * xtest_c[class_num].shape[1])
            # Get all the data points excluding the data point of a particular class (remove all the data for the current class)
            xtest_ood =  np.setdiff1d(a1_rows, a2_rows).view(ftest.dtype).reshape(-1, ftest.shape[1])

            # Obtain the test thresholds for this class where the data points of this particular class is removed from the rest
            class_test_ood_thresholds = self.get_class_thresholds(xtest_ood, means[class_num], cov[class_num],entropy[class_num])
            test_ood_thresholds.append(class_test_ood_thresholds)

        return test_ood_thresholds
   
    def get_online_ood_thresholds(self, means, cov, entropy, food):
        # Treating other classes as OOD dataset for a particular class
        ood_thresholds = [] # List of all threshold values
        # All the data for the different classes of the test features

        # Iterate through the classes
        for class_num in range(len(means)):
            class_ood_thresholds = self.get_class_thresholds(food,means[class_num], cov[class_num],entropy[class_num])
            ood_thresholds.append(class_ood_thresholds)
        
        return ood_thresholds

    # General function to ge the thresholds
    def get_class_thresholds(self,fdata,class_means,class_cov,class_entropy):
        class_thresholds = [] #  List of class threshold values
        # obtain the num batches
        num_batches = len(fdata)//self.typicality_bsz 
        
        for i in range(num_batches):
            fdata_batch = fdata[(i*self.typicality_bsz):((i+1)*self.typicality_bsz)]
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
    
    # Normalises the data
    def normalise(self,ftrain, fval, ftest,food):
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        fval /= np.linalg.norm(fval, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        fval = (fval - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        
        return ftrain, fval, ftest, food


    def get_eval_results(self, ftrain,fval, ftest, food, labelstrain, labelsval,labels_test):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, fval_norm, ftest_norm, food_norm = self.normalise(ftrain, fval, ftest, food)
        class_means, class_cov,class_entropy, class_quantile_thresholds = self.get_offline_thresholds(ftrain_norm, fval_norm, labelstrain, labelsval)
        # Class conditional thresholds for the correct class
        
        test_thresholds = self.get_online_test_thresholds(class_means,class_cov,class_entropy,ftest_norm,labels_test)
        # Class conditional thresholds using the incorrect class for the ID test data
        test_ood_thresholds = self.get_online_test_ood_thresholds(class_means,class_cov,class_entropy,ftest_norm,labels_test)
        # Class conditional thresholds using OOD test data
        ood_thresholds = self.get_online_ood_thresholds(class_means, class_cov,class_entropy, food_norm)
        
        
        
        #import ipdb; ipdb.set_trace()
        ######
        self.OVR_AUROC_saving(test_thresholds,ood_thresholds,f'Class vs OOD Rest {self.OOD_dataname}',f'Typicality One Vs OOD Rest {self.OOD_dataname}')
        #print('USING TYPICALITY FOR ood TE')
        '''
        ood_table_data = {f'Class vs OOD Rest {self.OOD_dataname}': [],'AUROC': []}
        for class_num in range(len(test_thresholds)):
            ood_table_data[f'Class vs OOD Rest {self.OOD_dataname}'].append(class_num)
            class_ood_auroc = get_roc_sklearn(test_thresholds[class_num], ood_thresholds[class_num])
            ood_table_data['AUROC'].append(round(class_ood_auroc,2)) # Append the value rounded to 2 decimal places

        ood_table_df = pd.DataFrame(ood_table_data)
        ood_table = wandb.Table(dataframe=ood_table_df)
        wandb.log({f"Typicality One Vs OOD Rest {self.OOD_dataname}": ood_table})
        table_saving(ood_table_df,f"Typicality One Vs OOD Rest {self.OOD_dataname}")    
        '''

        
        ######
        # OVR for the case of test data against over classes of test data
        #self.OVR_AUROC_saving(test_thresholds,test_ood_thresholds,'Class vs Rest','Typicality One vs Rest')
        '''
        table_data = {'Class vs Rest': [],'AUROC': []}
        for class_num in range(len(test_thresholds)):
            table_data['Class vs Rest'].append(class_num)
            #import ipdb; ipdb.set_trace()
            #import ipdb; ipdb.set_trace() 
            class_auroc = get_roc_sklearn(test_thresholds[class_num], test_ood_thresholds[class_num])
            table_data['AUROC'].append(round(class_auroc,2)) # Append the value rounded to 2 decimal places

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({"Typicality One Vs Rest": table})
        table_saving(table_df,'Typicality One vs Rest')
        '''
    def OVR_AUROC_saving(self,test_thresholds,ood_thresholds,table_name,wandb_name):
        table_data = {table_name: [],'AUROC': []}
        for class_num in range(len(test_thresholds)):
            table_data[table_name].append(class_num)
            auroc = get_roc_sklearn(test_thresholds[class_num], ood_thresholds[class_num])
            table_data['AUROC'].append(round(auroc,2)) # Append the value rounded to 2 decimal places

        table_df = pd.DataFrame(table_data)
        table = wandb.Table(dataframe=table_df)
        wandb.log({wandb_name: table})
        import ipdb; ipdb.set_trace()
        table_saving(table_df,wandb_name) 

        

class Typicality_OVO(Typicality_OVR):
    def __init__(self, Datamodule,OOD_Datamodule,
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True,
        bootstrap_num: int = 50,
        typicality_bsz:int = 25):
        
        super().__init__(Datamodule, OOD_Datamodule, vector_level=vector_level,
        label_level=label_level,
        quick_callback=quick_callback,
        bootstrap_num=bootstrap_num,
        typicality_bsz=typicality_bsz)
    
    def get_ovo_test_ood_thresholds(self, means, cov, entropy, ftest, ytest):
        test_ood_thresholds = []

        xtest_c = [ftest[ytest == i] for i in np.unique(ytest)]

        for j in range(len(np.unique(ytest))):
            ovo_ood_thresholds = []
            for k in range(len(np.unique(ytest))):
                # Gives comparison between class j and class k
                
                class_test_ood_thresholds = self.get_class_thresholds(xtest_c[k], means[j], cov[j],entropy[j])
                # ovo is a list of lists
                ovo_ood_thresholds.append(class_test_ood_thresholds)
            # test_ood_thresholds is a list of lists of lists which contains all the values
            test_ood_thresholds.append(ovo_ood_thresholds)

        return test_ood_thresholds
    
    def get_eval_results(self, ftrain,fval, ftest, food, labelstrain, labelsval,labels_test):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
        
        # Nawid - obtain the scores for the test data and the OOD data
        ftrain_norm, fval_norm, ftest_norm, food_norm = self.normalise(ftrain, fval, ftest, food)
        class_means, class_cov,class_entropy, class_quantile_thresholds = self.get_offline_thresholds(ftrain_norm, fval_norm, labelstrain, labelsval)
        # Class conditional thresholds for the correct class
        
        test_thresholds = self.get_online_test_thresholds(class_means,class_cov,class_entropy,ftest_norm,labels_test)
        ovo_test_thresholds = self.get_ovo_test_ood_thresholds(class_means,class_cov,class_entropy,ftest_norm,labels_test)

        column_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        index_names = [f'{i}' for i in range(len(np.unique(labels_test)))]
        data = np.zeros((len(index_names),len(column_names)))

        for i in range(len(np.unique(labels_test))):
            for j in range(len(np.unique(labels_test))):
                class_auroc = get_roc_sklearn(test_thresholds[i], ovo_test_thresholds[i][j])               
                data[i,j] = class_auroc

        data = np.around(data,decimals=1)
        table_df = pd.DataFrame(data, index = index_names, columns=column_names)
        # plot heat with annotations of the value as well as formating to 2 decimal places          

        # Choose whether to show annotations based on the number of examples present
        if len(np.unique(labels_test)) >10: 
            sns.heatmap(table_df,annot=False,fmt=".1f")
        else:
            sns.heatmap(table_df,annot=True,fmt=".1f")

        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Typicality One vs One Confusion Matrix')
        ovo_filename = f'ovo_conf_matrix.png'
        plt.savefig(ovo_filename,bbox_inches='tight')
        wandb_ovo = f'Typicality One vs One Matrix'    
        wandb.log({wandb_ovo:wandb.Image(ovo_filename)})
        # Update the data table to selectively remove different tables of the data
        row_values = [j for j in range(len(np.unique(labels_test)))]
        updated_data = np.insert(data,0,values = row_values, axis=1)
        column_names.insert(0,'Class') # Inplace operation
        updated_table_df = pd.DataFrame(updated_data, columns=column_names)

        table = wandb.Table(dataframe=updated_table_df)

        wandb.log({"Typicality One Vs One": table})
        # NEED TO CLOSE OTHERWISE WILL HAVE OVERLAPPING MATRICES SAVED IN WANDB
        plt.close()