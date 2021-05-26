import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_fpr,\
    get_pr_sklearn, get_roc_sklearn

import faiss
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
import wandb
import sklearn.metrics as skm
import faiss

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean



class Mahalanobis_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,num_inference_clusters,quick_callback):
        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.num_inference_clusters = num_inference_clusters
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
        self.log_name = "Supervised_Mahalanobis_"
        self.unsupervised_log_name = "Unsupervised_Mahalanobis_"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'
        
        self.log_name = "Mahalanobis_"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'
        
        
        # Number of classes of the data
        self.num_hierarchy = self.Datamodule.num_hierarchy  # Used to get the number of layers in hierarchy
        self.num_fine_classes = self.Datamodule.num_classes
        self.num_coarse_classes = self.Datamodule.num_coarse_classes
        # Names for creating a confusion matrix for the data
        class_dict = self.Datamodule.idx2class
        self.class_names = [v for k,v in class_dict.items()] # names of the categories of the dataset
        # Obtain class names list for the case of the OOD data
        OOD_class_dict = self.OOD_Datamodule.idx2class
        self.OOD_class_names = [v for k,v in OOD_class_dict.items()] # names of the categories of the dataset

        self.OOD_dataname = self.OOD_Datamodule.name
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if fast testing as this can lead to issues with the code
        if trainer.fast_dev_run:
            pass
        else:
            self.forward_callback(trainer=trainer, pl_module=pl_module) 
    
    def on_test_epoch_end(self,trainer,pl_module):
        if trainer.fast_dev_run:
            pass
        else:
            self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        # 
        # Obtain representations of the data
        features_train, labels_train = self.get_features_hierarchy(pl_module, train_loader)
        #features_train, fine_labels_train, coarse_labels_train = self.get_features_hierarchy(pl_module, train_loader)  # using feature befor MLP-head
        features_test, labels_test = self.get_features_hierarchy(pl_module, test_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        #features_test, fine_labels_test, coarse_labels_test = self.get_features_hierarchy(pl_module, test_loader)
        #features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        import ipdb; ipdb.set_trace()
        for i in range(len(features_train)):
            # Obtain the number of classes for the case for the coarse and fine case
            num_classes = max(labels_train[i]+1)
            fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results(
                np.copy(features_train[i]),
                np.copy(features_test[i]),
                np.copy(features_ood[i]),
                np.copy(labels_train[i]),
                num_classes) 
            #import ipdb; ipdb.set_trace()
            self.mahalanobis_classification(indices_dtest, labels_test[i],f'Mahalanobis Classification {i+1}')

        # Calculates the mahalanobis distance using unsupervised approach
        
        # Perform unsupervised clustering with different values       
        for num_clusters in self.num_inference_clusters:
            _, _, _, UL_indices_dtest, UL_indices_dood = self.get_eval_results(
            np.copy(features_train[0]),
            np.copy(features_test[0]),
            np.copy(features_ood[0]),
            None,
            num_clusters
        )
             
        # input the ood predictions as well as the OOD labels to see how it performs
        
        return fpr95,auroc,aupr 

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels,name):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        #import ipdb; ipdb.set_trace()       
        wandb.log({name:mahalanobis_test_accuracy})

    def get_features_hierarchy(self,pl_module, dataloader, max_images=10**10, verbose=False):
        #features, fine_labels, coarse_labels = [], [], []
        '''
        sample = next(iter(dataloader)) 
        data_sample, *label_sample, index_sample = sample
        # Examines whether it is only fine labels, or fine and coarse (required as ID dataloader and OOD dataloader differs)
        hierarchy_layers = len(label_sample)
        '''
        features = {}
        labels = {}

        for i in range(self.num_hierarchy):
            features[i] = []
            labels[i] = []
        
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, *label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            if total > max_images:
                break
            
            img = img.to(pl_module.device)
            #import ipdb; ipdb.set_trace()

            feature_vector = pl_module.callback_vector(img)
            for i in range(self.num_hierarchy):
                features[i] += list(feature_vector[i].data.cpu().numpy())
                labels[i] += list(label[i].data.cpu().numpy())

            
            if verbose and not index % 50:
                print(index)
                
            total += len(img)
        
        # Convert each list separately to an array
        for i in range(self.num_hierarchy):
            features[i] = np.array(features[i])
            labels[i] = np.array(labels[i])
        
        return features, labels
    
    def get_features(self,pl_module, dataloader, max_images=10**10, verbose=False):
        #features, fine_labels, coarse_labels = [], [], []
        features = {}
        labels = []

        for i in range(self.num_hierarchy):
            features[i] = []
        
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img,label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            if total > max_images:
                break
            
            img = img.to(pl_module.device)
            feature_vector = pl_module.callback_vector(img)
            for i in range(self.num_hierarchy):
                features[i] += list(feature_vector[i].data.cpu().numpy())
            #import ipdb; ipdb.set_trace()
            labels += list(label.data.cpu().numpy())

            
            if verbose and not index % 50:
                print(index)
                
            total += len(img)
        
        # Convert each list separately to an array
        for i in range(self.num_hierarchy):
            features[i] = np.array(features[i])
        
        labels = np.array(labels)
        
        return features, labels   

    # Nawid - perform k-means on the training features and then assign a cluster assignment
    def get_clusters(self, ftrain, nclusters):
        kmeans = faiss.Kmeans(
            ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=True
        )
        kmeans.train(np.random.permutation(ftrain))
        _, ypred = kmeans.assign(ftrain)
        return ypred
    
    def get_scores(self,ftrain, ftest, food, labelstrain,n_clusters):
        if labelstrain is None:
            ypred = self.get_clusters(ftrain=ftrain, nclusters=n_clusters)
        else:
            ypred = labelstrain
        return self.get_scores_multi_cluster(ftrain, ftest, food, ypred)
    
    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
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
    # Used to obtain cluster centroids of the data
    def get_cluster_centroids(self,ftrain,ypred):
        centroids = []
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        centroids = np.array([np.mean(i, axis=0) for i in xc])
        
        
        #centroids = [np.expand_dims(np.mean(i, axis=0),0) for i in xc]  # Nawid - find the average of the feature vectors of each class
        return centroids

    # Used to calculate the distances from centroids
    def centroid_distances(self,ftrain,ypred):
        # Makes barchart of deviations from the average vector of the training set
        avg_vector = np.mean(ftrain, axis=0)
        avg_vector = np.reshape(avg_vector, (1, -1))
        centroids = self.get_cluster_centroids(ftrain,ypred)
        diff = np.abs(centroids - avg_vector) # Calculates the absolute difference element wise to ensure that the mean does not cancel out
        total_diff = np.mean(diff, axis=1)
        labels = [i for i in np.unique(ypred)] 
        data =[[label, val] for (label ,val) in zip(labels,total_diff)] # iterates through the different labels as well as the different values for the labels
        table = wandb.Table(data=data, columns = ["label", "value"])
        wandb.log({"Centroid Distances Average vector" : wandb.plot.bar(table, "label", "value",
                               title="Centroid Distances Average vector")})
        

        # Makes barchart of deviations from the vector of all zeros where I assume the centre of the hypersphere is a vector of zeros
        zero_vector = np.zeros_like(avg_vector)
        zero_diff = np.abs(centroids - zero_vector) # Calculates the absolute difference element wise to ensure that the mean does not cancel out
        total_zero_diff = np.mean(zero_diff, axis=1)
        data_zero =[[label, val_zero] for (label ,val_zero) in zip(labels,total_zero_diff)] # iterates through the different labels as well as the different values for the labels
        table_zero = wandb.Table(data=data_zero, columns = ["label", "value"])

        wandb.log({"Centroid Distances Zero vector" : wandb.plot.bar(table_zero, "label", "value",
                               title="Centroid Distances Zero vector")})
        
        #import ipdb; ipdb.set_trace()
        #return total_diff
    

        
    def get_eval_results(self,ftrain, ftest, food, labelstrain, num_clusters):
        """
            None.
        """
        #import ipdb; ipdb.set_trace()
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
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain,num_clusters)
        self.log_confidence_scores(dtest,dood,labelstrain,num_clusters)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
        
        if labelstrain is None:
            wandb.log({self.unsupervised_log_name + f'AUROC_{num_clusters}_clusters_{self.OOD_dataname}': auroc})
        else:
            wandb.log({self.log_name + f'AUROC_{num_clusters}_clusters_{self.OOD_dataname}': auroc})
        return fpr95, auroc, aupr, indices_dtest, indices_dood
        
    
    def distance_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +f"conf_mat_id_{self.OOD_dataname}": wandb.plot.confusion_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names),
            "global_step": trainer.global_step
                  })
    '''
    def distance_OOD_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +"OOD_conf_mat_id": OOD_conf_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names,OOD_class_names =self.OOD_class_names),
            "global_step": trainer.global_step
                  })
    '''
    def supervised_distance_OOD_confusion_matrix(self,trainer,predictions,labels):
        # Used for the case of EMNIST where the number of labels differs compared to the OOD class
        if len(self.class_names) == len(self.OOD_class_names):
            wandb.log({self.log_name +f"OOD_conf_mat_id_supervised_{self.OOD_dataname}": OOD_conf_matrix(probs = None,
                preds=predictions, y_true=labels,
                class_names=self.class_names,OOD_class_names =self.OOD_class_names),
                "global_step": trainer.global_step
                      })
        else:
            wandb.log({self.log_name +f"OOD_conf_mat_id_supervised_{self.OOD_dataname}": OOD_conf_matrix(probs = None,
                preds=predictions, y_true=labels,
                class_names=None,OOD_class_names=None),
                "global_step": trainer.global_step
                      })
        
    
    def unsupervised_distance_OOD_confusion_matrix(self,trainer,num_clusters,predictions,labels):
        wandb.log({f'{self.log_name}_OOD_conf_mat_id_unsupervised:{num_clusters}_clusters_{self.OO_dataname}': OOD_conf_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=None,OOD_class_names =None),
            "global_step": trainer.global_step
                  })
    
    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,Dtest,DOOD,labels_train,num_clusters):  
        confidence_test = Dtest
        confidence_OOD  = DOOD
         # histogram of the confidence scores for the true data
        true_data = [[s] for s in confidence_test]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        # Examine if the centroid was obtained in supervised or unsupervised manner
        if labels_train is not None:    
            true_histogram_name = self.true_histogram + f':Supervised_{num_clusters}_clusters'
            ood_histogram_name = self.ood_histogram + f':Supervised_{num_clusters}_clusters_{self.OOD_dataname}'
        else:
            true_histogram_name = self.true_histogram + f':Unsupervised_{num_clusters}_clusters_{self.OOD_dataname}'
            ood_histogram_name = self.ood_histogram + f':Unsupervised_{num_clusters}_clusters_{self.OOD_dataname}'

        #import ipdb; ipdb.set_trace()
        wandb.log({true_histogram_name: wandb.plot.histogram(true_table, "scores",title=true_histogram_name)})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in confidence_OOD]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({ood_histogram_name: wandb.plot.histogram(ood_table, "scores",title=ood_histogram_name)})

def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

# calculates aupr
def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr

# Nawid - calculate false positive rate
def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)
