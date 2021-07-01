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

class Hierarchical_Mahalanobis(pl.Callback):
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

        self.true_histogram = 'Hierarchical Mahalanobis True data scores'
        self.ood_histogram = 'Hierarchical Mahalanobis OOD data scores'
    
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

        #import ipdb; ipdb.set_trace()
        # Number of classes obtained from the max label value + 1 ( to take into account counting from zero)
        
        _, _, _, dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))

        fpr95, auroc, aupr, dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            np.copy(indices_dtest_coarse),
            np.copy(indices_dood_coarse))
    
        test_accuracy = self.mahalanobis_classification(indices_dtest_fine, labels_test_fine)
        name = f'Mahalanobis Hierarchical Classification'
        wandb.log({name: test_accuracy})
        if trainer.testing:
            get_roc_plot(dtest_fine,dood_fine, f'Hierarchical_{self.OOD_dataname}', f'Hierarchical {self.OOD_dataname}')  
        return fpr95,auroc,aupr

    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        return mahalanobis_test_accuracy
    
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

    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,Ddata,histogram_name):  
        confidence_data = Ddata
         # histogram of the confidence scores for the true data
        data_scores = [[s] for s in confidence_data]
        table = wandb.Table(data=data_scores, columns=["scores"])
        # Examine if the centroid was obtained in supervised or unsupervised manner
            
       
        #import ipdb; ipdb.set_trace()
        wandb.log({histogram_name: wandb.plot.histogram(table, "scores",title=histogram_name)})

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
        
        if ptest_index is not None:
            true_histogram_name = 'Hierarchical Mahalanobis True data scores'
            ood_histogram_name = f'Hierarchical Mahalanobis OOD data scores {self.OOD_dataname}'
            self.log_confidence_scores(dtest,true_histogram_name)
            self.log_confidence_scores(dood,ood_histogram_name)
            
            fpr95 = get_fpr(dtest, dood)
            auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
            wandb.log({'Hierarchical Mahalanobis'+ f' AUROC: {self.OOD_dataname}': auroc})
            wandb.log({'Hierarchical Mahalanobis'+ f' AUPR: {self.OOD_dataname}': aupr})
            wandb.log({'Hierarchical Mahalanobis'+ f' FPR: {self.OOD_dataname}': fpr95})
        else:
            fpr95, auroc, aupr = None, None, None

        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood


class Hierarchical_scores_comparison(Hierarchical_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__(Datamodule,OOD_Datamodule,quick_callback)
        #print('Hierarchical being used')
    def forward_callback(self, trainer, pl_module):
        #print('Hierarchical being used')
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

        _, _, _, dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine))


        _, _, _, dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse,),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))


        _, _, _, dtest_conditional_fine, dood_conditional_fine, indices_dtest_conditional_fine, indices_dood_conditional_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            np.copy(indices_dtest_coarse),
            np.copy(indices_dood_coarse))


        #import ipdb; ipdb.set_trace()
        limit = min(len(dtest_fine),len(dood_fine))
          
        dtest_fine = dtest_fine[:limit]
        dtest_conditional_fine = dtest_conditional_fine[:limit]
        dood_fine = dood_fine[:limit]
        dood_conditional_fine = dood_conditional_fine[:limit]

        ID_dict = {'ID Fine': dtest_fine, 'ID Conditional Fine': dtest_conditional_fine}
        OOD_dict = {f'{self.OOD_dataname} Fine': dood_fine,f'{self.OOD_dataname} Conditional Fine': dood_conditional_fine}
        # https://towardsdatascience.com/merge-dictionaries-in-python-d4e9ce137374
        #all_dict = {**ID_dict,**OOD_dict} # Merged dictionary
        all_dict = {'ID Fine': dtest_fine, 'ID Conditional Fine': dtest_conditional_fine,f'{self.OOD_dataname} Fine': dood_fine,f'{self.OOD_dataname} Conditional Fine': dood_conditional_fine}
        #import ipdb; ipdb.set_trace()
        # Plots the counts, probabilities as well as the kde
        ID_name = f'Hierarchical Fine ID {self.OOD_dataname} data scores'
        OOD_name = f'Hierarchical Fine OOD {self.OOD_dataname} data scores'
        all_name = f'Hierarchical Fine All {self.OOD_dataname} data scores'

        # Replace white spaces with underscore  https://stackoverflow.com/questions/1007481/how-do-i-replace-whitespaces-with-underscore
        kde_plot(ID_dict,ID_name,ID_name.replace(" ","_"),ID_name)
        kde_plot(OOD_dict,OOD_name,OOD_name.replace(" ","_"),OOD_name)
        kde_plot(all_dict,all_name, all_name.replace(" ","_"),all_name)

        # Plots the table data for the model
        table_df = pd.DataFrame(all_dict)
        table = wandb.Table(data=table_df)
        wandb.log({all_name:table})
        

        '''
        # Logs the difference in improvement for the network
        self.conditional_accuracy_difference(indices_dtest_fine,indices_dtest_conditional_fine,labels_test_fine)
        
        self.joint_mahalanobis_classification(indices_dtest_coarse,labels_test_coarse,indices_dtest_fine,labels_test_fine)
        '''
        
    # Calculates conditional accuracy for the data
    def conditional_accuracy_difference(self, unconditional_pred, conditional_pred, labels):
        fine_unconditional_accuracy = self.mahalanobis_classification(unconditional_pred, labels)
        fine_conditional_accuracy = self.mahalanobis_classification(conditional_pred,labels)
        conditional_diff = fine_conditional_accuracy - fine_unconditional_accuracy
        
        wandb.run.summary['Fine Unconditional Accuracy'] = fine_unconditional_accuracy
        wandb.run.summary['Fine Conditional Accuracy'] = fine_conditional_accuracy
        wandb.run.summary['Fine Conditional Improvement'] = conditional_diff

    
    def joint_mahalanobis_classification(self,coarse_predictions, coarse_labels, fine_predictions,fine_labels):
        coarse_predictions = torch.tensor(coarse_predictions,dtype = torch.long)
        coarse_labels = torch.tensor(coarse_labels, dtype = torch.long)
        coarse_results = coarse_predictions.eq(coarse_labels)

        fine_predictions = torch.tensor(fine_predictions, dtype = torch.long)
        fine_labels = torch.tensor(fine_labels, dtype = torch.long)
        fine_results = fine_predictions.eq(fine_labels)

        coarse_correct_fine_correct = (coarse_results>0) & (fine_results >0) 
        coarse_correct_fine_incorrect = (coarse_results > 0) & (fine_results < 1)
        coarse_incorrect_fine_correct = (coarse_results < 1) & (fine_results > 0)
        coarse_incorrect_fine_incorrect  = (coarse_results < 1) & (fine_results < 1)
                
        coarse_correct_fine_correct = (100*sum(coarse_correct_fine_correct) /len(coarse_predictions)).item() # change from tensor to scalar
        coarse_correct_fine_incorrect = (100*sum(coarse_correct_fine_incorrect) /len(coarse_predictions)).item()
        coarse_incorrect_fine_correct = (100*sum(coarse_incorrect_fine_correct) /len(coarse_predictions)).item()
        coarse_incorrect_fine_incorrect = (100*sum(coarse_incorrect_fine_incorrect) /len(coarse_predictions)).item()

        table_data = {'Coarse Correct Fine Correct (%)':[],'Coarse Correct Fine Incorrect (%)':[],'Coarse Incorrect Fine Correct (%)':[],'Coarse Incorrect Fine Incorrect (%)':[]}
        table_data['Coarse Correct Fine Correct (%)'].append(coarse_correct_fine_correct)
        table_data['Coarse Correct Fine Incorrect (%)'].append(coarse_correct_fine_incorrect)
        table_data['Coarse Incorrect Fine Correct (%)'].append(coarse_incorrect_fine_correct)
        table_data['Coarse Incorrect Fine Incorrect (%)'].append(coarse_incorrect_fine_incorrect)

        table_df = pd.DataFrame(table_data)
    
        table = wandb.Table(dataframe=table_df)
        wandb.log({f"Joint Coarse Fine classification": table})
        table_saving(table_df,'Joint Mahalanobis Classification')


    def get_eval_results(self,ftrain, ftest, food, labelstrain,ptest_index = None, pood_index=None):
        if ptest_index is not None:
            assert pood_index is not None, 'conditioning on the test data but not on OOD should not occur'
        """
            None.
        """
        
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, ptest_index, pood_index)
        
        fpr95, auroc, aupr = None, None, None

        return fpr95, auroc, aupr, dtest, dood, indices_dtest, indices_dood





class Hierarchical_Subsample(Hierarchical_Mahalanobis):
    def __init__(self, Datamodule,OOD_Datamodule,
        quick_callback:bool = True,
        bootstrap_num: int = 25):
    
        super().__init__(Datamodule,OOD_Datamodule,quick_callback)

        # Get the number of coarse classes
        self.num_coarse = self.Datamodule.num_coarse_classes
        # Number of times to bootstrap sample the data
        self.bootstrap_num = bootstrap_num
        

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


        # Used to calculate coarse accuracy using the coarse representations
        dtest_coarse, dood_coarse, indices_dtest_coarse, indices_dood_coarse = self.get_eval_results(
            np.copy(features_train_coarse),
            np.copy(features_test_coarse,),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))
        
        
        # Used to caclulate coarse accuracy using the fine representations
        dtest_fine, dood_fine, indices_dtest_fine, indices_dood_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_coarse))

        coarse_coarse_test_accuracy = self.mahalanobis_classification(indices_dtest_coarse, labels_test_coarse).item()
        fine_coarse_test_accuracy = self.mahalanobis_classification(indices_dtest_fine, labels_test_coarse).item()

        # Look at the classes available
        classes_available = len(np.unique(labels_train_fine))
        accuracy_values = []
        for i in range(self.bootstrap_num):
            # Investigate the different classes
            specific_classes = np.random.choice(classes_available, size=self.num_coarse, replace=False) # Obtain a sub sample of values

            class_train_masks = [labels_train_fine==specific_class for specific_class in specific_classes] # Obtain masks for class
            train_mask = np.sum(class_train_masks,axis=0) > 0 # Make a joint mask for all te different masks
            specific_labels_train = labels_train_fine[train_mask] # Obtain labels for the different data points
            specific_features_train =  features_train_fine[train_mask] # Nawid - 
            # Test data            
            class_test_masks = [labels_test_fine==specific_class for specific_class in specific_classes]
            test_mask = np.sum(class_test_masks,axis=0) > 0
            specific_labels_test = labels_test_fine[test_mask] # Obtain certain labels
            specific_features_test = features_test_fine[test_mask] # obtain datapoints which belong to certain classes

            # Sort the values for the specific class
            sorted_classes = sorted(specific_classes)
            
            features_ood_fine = features_ood_fine[0:32] # shorten the data of OOD as this is not actually important for the ovo classifier
            # Map the subsamples labels to values between 0 and n where n is the number of coarse labels used 
            for j in range(len(specific_classes)):
                #import ipdb; ipdb.set_trace()
                specific_labels_train[specific_labels_train == sorted_classes[j]] = j
                specific_labels_test[specific_labels_test == sorted_classes[j]] = j 
            
            # Obtain subs sample predictions for the fine labels
            dtest, dood, indices_dtest, indices_dood = self.get_eval_results(
                    np.copy(specific_features_train),
                    np.copy(specific_features_test),
                    np.copy(features_ood_fine),
                    np.copy(specific_labels_train))
            
            test_accuracy = self.mahalanobis_classification(indices_dtest, specific_labels_test).item()
            accuracy_values.append(test_accuracy)
        
        
        # Calculate statistics related to the value
        table_data =  {'Coarse-Coarse (%)':[],'Fine-Coarse (%)':[],'Subsample Mean (%)':[],'Subsample Std (%)':[],'Subsample Min (%)':[], 'Subsample Max (%)':[]}
        table_data['Coarse-Coarse (%)'].append(coarse_coarse_test_accuracy)
        table_data['Fine-Coarse (%)'].append(fine_coarse_test_accuracy)

        table_data['Subsample Mean (%)'].append(statistics.mean(accuracy_values))
        table_data['Subsample Std (%)'].append(statistics.stdev(accuracy_values))
        table_data['Subsample Min (%)'].append(min(accuracy_values))
        table_data['Subsample Max (%)'].append(max(accuracy_values))

        table_df = pd.DataFrame(table_data)
        
        table_saving(table_df,'Fine Grain Subsampling')

        wandb.run.summary['Coarse-Coarse Accuracy (%)'] = coarse_coarse_test_accuracy
        wandb.run.summary['Fine-Coarse Accuracy (%)'] = fine_coarse_test_accuracy 
        wandb.run.summary['Fine Subsample Mean Accuracy (%)'] = statistics.mean(accuracy_values)
        # NEED TO CLOSE OTHERWISE WILL HAVE OVERLAPPING MATRICES SAVED IN WANDB

    
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain)
        return dtest, dood, indices_dtest, indices_dood


class Hierarchical_Relative_Mahalanobis(Hierarchical_Mahalanobis):
    def __init__(self, Datamodule, OOD_Datamodule,
        quick_callback:bool = True):

        super().__init__(Datamodule, OOD_Datamodule, quick_callback)
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    # Performs all the computation in the callback
    def forward_callback(self, trainer, pl_module):
        #print('HIERARCHICAL RELATIVE BEING USED')
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
            np.copy(features_train_coarse),
            np.copy(features_test_coarse),
            np.copy(features_ood_coarse),
            np.copy(labels_train_coarse))

        dtest_conditional_fine, dood_conditional_fine, indices_dtest_conditional_fine, indices_dood_conditional_fine = self.get_eval_results(
            np.copy(features_train_fine),
            np.copy(features_test_fine),
            np.copy(features_ood_fine),
            np.copy(labels_train_fine),
            np.copy(indices_dtest_coarse),
            np.copy(indices_dood_coarse))
        
        # AUROC for the different situations
        self.relative_hierarchical_AUROC(dtest_conditional_fine, dood_conditional_fine,dtest_fine, dood_fine,
                        f'Hierarchical_Relative_Mahalanobis AUROC: {self.OOD_dataname}',f'Hierarchical Relative Mahalobis AUROC Improvement:{self.OOD_dataname}')
        
        # Classification improvement
        self.relative_hierarchical_classification(indices_dtest_conditional_fine,indices_dtest_fine,labels_test_fine,
            f'Hierarchical Relative Mahalanobis Classification','Hierarchical Relative Mahalobis Classification Improvement')
        
        # Plotting the confidence scores for the situation
        self.relative_hiearachical_scores_saving(dtest_conditional_fine,dood_conditional_fine,dtest_fine,dood_fine,f' Hierarchical Relative Mahalanobis - {self.OOD_dataname} data scores')


    # Calaculates the accuracy of a data point based on the closest distance to a centroid
    def mahalanobis_classification(self,predictions, labels):
        predictions = torch.tensor(predictions,dtype = torch.long)
        labels = torch.tensor(labels,dtype = torch.long)
        mahalanobis_test_accuracy = 100*sum(predictions.eq(labels)) /len(predictions) 
        return mahalanobis_test_accuracy
    
    def relative_hierarchical_AUROC(self,din_conditional_fine,dood_conditional_fine,din_fine,dood_fine, name, improvement_name):
        # AUROC for the different situations
        hierarchical_auroc = get_roc_sklearn(din_conditional_fine, dood_conditional_fine)
        auroc =  get_roc_sklearn(din_fine, dood_fine)
        hierarchical_auroc_improvement = hierarchical_auroc - auroc
        wandb.run.summary[name]= hierarchical_auroc
        wandb.run.summary[improvement_name] = hierarchical_auroc_improvement
    
    def relative_hierarchical_classification(self,din_conditional_fine,din_fine,labels, name, improvement_name):
        hierarchical_test_accuracy = self.mahalanobis_classification(din_conditional_fine, labels)
        unconditional_test_accuracy = self.mahalanobis_classification(din_fine,labels)
        classification_improvement =hierarchical_test_accuracy - unconditional_test_accuracy
        wandb.run.summary[name] =  hierarchical_test_accuracy
        wandb.run.summary[improvement_name] = classification_improvement 
    
    def relative_hiearachical_scores_saving(self, din_conditional_fine, dood_conditional_fine, din_fine, dood_fine,data_name):
        # Plotting the confidence scores for the situation
        # Saves the confidence valeus of the data table
        limit = min(len(din_fine),len(dood_fine))
        din_fine = din_fine[:limit]
        dood_fine = dood_fine[:limit]
        din_conditional_fine = din_conditional_fine[:limit]
        dood_conditional_fine = dood_conditional_fine[:limit]

        # https://towardsdatascience.com/merge-dictionaries-in-python-d4e9ce137374
        data_dict = {f'ID Fine': din_fine, f'ID Conditional Fine': din_conditional_fine,
            f'OOD Fine {self.OOD_dataname}':dood_fine, f'OOD Conditional Fine {self.OOD_dataname}':dood_conditional_fine}
        # Plots the counts, probabilities as well as the kde
        table_df = pd.DataFrame(data_dict)
        table = wandb.Table(data=table_df)
        wandb.log({data_name:table})
    
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

        # Calculating background statistics
        mean = np.mean(ftrain, axis=0,keepdims=True)
        cov = np.cov(ftrain.T, bias=True)
        
        background_din = np.sum(
                ftest - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (ftest- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        background_dood = np.sum(
                food - mean
                * (
                    np.linalg.pinv(cov).dot(
                        (food- mean).T
                    )
                ).T,
                axis=-1,
            )
        
        din = din - background_din
        dood = dood - background_dood
        
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
        if ptest_index is not None:
            assert pood_index is not None, 'conditioning on the test data but not on OOD should not occur'
        """
            None.
        """

        ftrain_norm, ftest_norm, food_norm = self.normalise(ftrain,ftest,food)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain_norm, ftest_norm, food_norm, labelstrain, ptest_index, pood_index)
        
        return dtest, dood, indices_dtest, indices_dood



def kde_plot(input_data,title_name,file_name,wandb_name):
    sns.displot(data =input_data,fill=False,common_norm=False,kind='kde', multiple="stack")
    plt.xlabel('Distance')
    plt.ylabel('Normalized Density')
    plt.xlim([0, 1000])
    plt.title(f'{title_name}')
    kde_filename = f'Images/{file_name}.png'
    plt.savefig(kde_filename,bbox_inches='tight')
    plt.close()
    wandb_distance = f'{wandb_name}'
    wandb.log({wandb_distance:wandb.Image(kde_filename)})

def count_plot(input_data,title_name,file_name,wandb_name):
    sns.displot(data =input_data,fill=True,common_norm=False, multiple="stack")#,binrange = (0,1000))
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    #plt.xlim([0, 1000])
    plt.title(f'{title_name}')
    filename = f'Images/{file_name}.png'
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    wandb_distance = f'{wandb_name}'
    wandb.log({wandb_distance:wandb.Image(filename)})

def get_roc_plot(xin, xood,filename,logname):
    anomaly_targets = [0] * len(xin)  + [1] * len(xood)
    outputs = np.concatenate((xin, xood))

    fpr, trp, thresholds = skm.roc_curve(anomaly_targets, outputs)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=fpr, y=trp,
    legend="full",
    alpha=0.3
    )
    # Set  x and y-axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    ROC_filename = f'Images/ROC_{filename}.png'
    plt.savefig(ROC_filename)
    wandb_ROC = f'ROC curve: OOD dataset {logname}'
    wandb.log({wandb_ROC:wandb.Image(ROC_filename)})
