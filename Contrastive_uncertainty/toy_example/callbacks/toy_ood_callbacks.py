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

#from Contrastive_uncertainty.general.utils import iso_forest as iso
import eif as iso
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()

'''
class OOD_ROC(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule

        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        #self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

    # Outputs the OOD scores only (does not output the curve)
    def on_validation_epoch_end(self,trainer,pl_module):
        accuracy, roc_auc = self.get_auroc_ood_basic(pl_module)

    # Outputs OOD scores aswell as the ROC curve
    def on_test_epoch_end(self,trainer,pl_module):
        accuracy, roc_auc = self.get_auroc_ood(pl_module)


    def prepare_ood_datasets(self): # Code seems to be same as DUQ implementation
        true_dataset,ood_dataset = self.Datamodule.test_dataset, self.OOD_Datamodule.test_dataset # Dataset is not transformed at this point
        datasets = [true_dataset, ood_dataset]

        # Nawid - targets for anomaly, 0 is not anomaly and 1 is anomaly
        anomaly_targets = torch.cat(
            (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
        )

        concat_datasets = torch.utils.data.ConcatDataset(datasets)
        # Dataset becomes transformed as it passes through the datalodaer I believe
        dataloader = torch.utils.data.DataLoader(
            concat_datasets, batch_size=100, shuffle=False, num_workers=6, pin_memory=False,
        )

        return dataloader, anomaly_targets # Nawid -contains the data for the true data and the false data aswell as values which indicate whether it is an anomaly or not

    def loop_over_dataloader(self,pl_module, dataloader): # Nawid - obtain accuracy (pred equal to target) and kernel distances which are the scores
        #model.eval()

        with torch.no_grad():
            scores = []
            accuracies = []
            outputs = []
            #loader = quickloading(self.quick_callback,dataloader) # Used to get a single batch or used to get the entire dataset
            for data, target,index in dataloader:
                if isinstance(data, tuple) or isinstance(data, list):
                    data, *aug_data = data # Used to take into accoutn whether the data is a tuple of the different augmentations

                data = data.to(pl_module.device)
                target = target.to(pl_module.device)

                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = pl_module.class_discrimination(data)
                kernel_distance, pred = output.max(1) # The higher the kernel distance is, the more confident it is, so the lower the probability of an outlier

                # Make array for the probability of being outlier and not being an outlier which is based on the confidence (kernel distance)
                OOD_prediction = torch.zeros((len(data),2)).to(pl_module.device)

                OOD_prediction[:, 0] = kernel_distance # The kernel distance is the confidence that the label is 0 (not outlier)
                OOD_prediction[:, 1] = 1 - OOD_prediction[:, 0] # Value of being an outlier (belonging to class 1), is 1 - kernel distance (1-confidence)

                accuracy = pred.eq(target)
                accuracies.append(accuracy.cpu().numpy())

                scores.append(-kernel_distance.cpu().numpy()) # Scores represent - kernel distance, since kernel distance is the confidence of being in domain, - kernel distance represent OOD samples
                outputs.append(OOD_prediction.cpu().numpy())

        scores = np.concatenate(scores)
        accuracies = np.concatenate(accuracies)
        outputs = np.concatenate(outputs)

        return scores, accuracies, outputs

    # Outputs the OOD values as well as the ROC curve
    def get_auroc_ood(self,pl_module):
        dataloader, anomaly_targets = self.prepare_ood_datasets() # Nawid - obtains concatenated true and false data, as well as anomaly targets which show if the value is 0 (not anomaly) or 1 (anomaly)

        scores, accuracies,outputs = self.loop_over_dataloader(pl_module, dataloader)

        confidence_scores = outputs[:,0]
        # confidence scores for the true data and for the OOD data
        true_scores =  confidence_scores[: len(self.Datamodule.test_dataset)]
        ood_scores =  confidence_scores[len(self.Datamodule.test_dataset):]

        # histogram of the confidence scores for the true data
        true_data = [[s] for s in true_scores]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({'True_data_OOD_scores': wandb.plot.histogram(true_table, "scores",title="true_data_scores")})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in ood_scores]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({'OOD_data_OOD_scores': wandb.plot.histogram(ood_table, "scores",title="OOD_data_scores")})

        
        # Double checking the calculation of the ROC scores for the data
        
        #second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        #wandb.log({"AUROC_v2":second_check_roc_auc})
        

        accuracy = np.mean(accuracies[: len(self.Datamodule.test_dataset)]) # Nawid - obtainn accuracy for the true dataset
        roc_auc = roc_auc_score(anomaly_targets, scores) # Nawid - obtain roc auc for a binary classification problem where the scores show how close it is to a centroid (which will say whether it is in domain or out of domain), and whether it identified as being in domain or out of domain.

        wandb.log({"roc" : wandb.plot.roc_curve(anomaly_targets, outputs,#scores,
                        labels=None, classes_to_plot=None)})

        wandb.log({"AUROC":roc_auc})

        return accuracy, roc_auc

    
    def get_auroc_ood_basic(self,pl_module):
        dataloader, anomaly_targets = self.prepare_ood_datasets() # Nawid - obtains concatenated true and false data, as well as anomaly targets which show if the value is 0 (not anomaly) or 1 (anomaly)

        scores, accuracies,outputs = self.loop_over_dataloader(pl_module, dataloader)

        confidence_scores = outputs[:,0]
        # confidence scores for the true data and for the OOD data
        true_scores =  confidence_scores[: len(self.Datamodule.test_dataset)]
        ood_scores =  confidence_scores[len(self.Datamodule.test_dataset):]

        # histogram of the confidence scores for the true data
        true_data = [[s] for s in true_scores]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({'True_data_OOD_scores': wandb.plot.histogram(true_table, "scores",title="true_data_scores")})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in ood_scores]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({'OOD_data_OOD_scores': wandb.plot.histogram(ood_table, "scores",title="OOD_data_scores")})

        
        # Double checking the calculation of the ROC scores for the data
        #second_check_scores = #1 + scores # scores are all negative values
        #second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        #wandb.log({"AUROC_v2":second_check_roc_auc})
        

        accuracy = np.mean(accuracies[: len(self.Datamodule.test_dataset)]) # Nawid - obtainn accuracy for the true dataset
        roc_auc = roc_auc_score(anomaly_targets, scores) # Nawid - obtain roc auc for a binary classification problem where the scores show how close it is to a centroid (which will say whether it is in domain or out of domain), and whether it identified as being in domain or out of domain.

        wandb.log({"AUROC":roc_auc})

        return accuracy, roc_auc
'''
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

    # Used to calculate the distances between vectors
    def centre_distances(self,ftrain,ypred):
        xc = [ftrain[ypred == i] for i in np.unique(ypred)]
        
        avg_vector = np.mean(ftrain, axis=0)
        avg_vector = np.reshape(avg_vector, (1, -1))
        zero_vector = np.zeros_like(avg_vector)
        # concatenate the avg vector as well as the zero vector
        test_vector = np.concatenate((avg_vector,zero_vector),axis=0) 

        din = [
            np.sum(
                (test_vector - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (test_vector - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot preduct by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]
        # a list is given where each entry in the list represents the scores of a particular class
        # In each entry of the list, there are n values, which correspond to the n vectors which I am testing
        
        # Get the first scalar in each class for the case of avg vector difference
        avg_vector_diff = [din[i][0] for i in range(len(din))]
        labels = [i for i in np.unique(ypred)] 
        data =[[label, val] for (label ,val) in zip(labels,avg_vector_diff)] # iterates through the different labels as well as the different values for the labels
        table = wandb.Table(data=data, columns = ["label", "value"])
        wandb.log({"Mahalanobis Centroid Distances Average vector" : wandb.plot.bar(table, "label", "value",
                               title="Mahalanobis Centroid Distances Average vector")})
        

        zero_vector_diff = [din[i][1] for i in range(len(din))] 
        data_zero =[[label, val_zero] for (label ,val_zero) in zip(labels,zero_vector_diff)] # iterates through the different labels as well as the different values for the labels
        table = wandb.Table(data=data_zero, columns = ["label", "value"])
        wandb.log({"Mahalanobis Centroid Distances Zero vector" : wandb.plot.bar(table, "label", "value",
                               title="Mahalanobis Centroid Distances Zero vector")})

    # Used for making a confusion matrix based on the squared distance between point
    def get_ood_euclidean_scores_multi_cluster(self,ftrain, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        dood = [
            np.sum(
                np.square(food - np.mean(x, axis=0, keepdims=True)),
                axis=-1,
            )
            for x in xc # Nawid- this calculates the score for all the OOD examples 
        ]
        # Calculate the indices corresponding to the values
        
        indices_dood = np.argmin(dood, axis=0)
        dood = np.min(dood, axis=0) # Nawid - calculate the minimum distance
        
        return dood, indices_dood

    def distance_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +"conf_mat_id": wandb.plot.confusion_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names),
            "global_step": trainer.global_step
                  })

    def distance_OOD_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +"OOD_conf_mat_id": OOD_conf_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names,OOD_class_names =self.OOD_class_names),
            "global_step": trainer.global_step
                  })

# https://github.com/sahandha/eif#The-Code (Based on code from this section)
class IsoForest(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,quick_callback):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

    def on_validation_epoch_end(self, trainer, pl_module):
        self.get_iforest_auroc_basic(trainer,pl_module)

    def on_test_epoch_end(self,trainer,pl_module):
        self.get_iforest_auroc(trainer,pl_module)
        '''
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
        
        features_train, labels_train = self.get_features(pl_module, train_loader)  # using feature befor MLP-head
        features_test, labels_test = self.get_features(pl_module, train_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        self.iforest_score(features_train, features_test, features_ood)
        # Obtain the fpr95, aupr, test predictions, OOD predictions (basic version does not log the confidence scores or the confusion matrices)
        '''


    def get_features(self,pl_module, dataloader, max_images=10**10, verbose=False):
        features, labels = [], []
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, label,indices) in enumerate(loader):
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            if total > max_images:
                break
            
            img, label = img.to(pl_module.device), label.to(pl_module.device)

            features += list(pl_module.feature_vector(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            if verbose and not index % 50:
                print(index)
                
            total += len(img)
        #import ipdb; ipdb.set_trace()
        # https://www.programcreek.com/python/example/21245/numpy.double converting to a double 
        return np.asarray(features,dtype=np.double), np.array(labels) 
    
    def get_all_features(self,trainer,pl_module):
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        features_train, labels_train = self.get_features(pl_module, train_loader)  # using feature befor MLP-head
        features_test, labels_test = self.get_features(pl_module, train_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        return features_train, features_test, features_ood

    
    def iforest_score_basic(self,features_train,features_test, features_ood):
        F = iso.iForest(features_train,ntrees=200, sample_size=256, ExtensionLevel=1) # Train the isolation forest on indomain data
        SN = F.compute_paths(X_in=features_test) # Nawid - compute the paths for the nominal datapoints
        SA = F.compute_paths(X_in=features_ood)
        iforest_auroc = get_roc_sklearn(SN, SA)
        wandb.log({'iForest AUROC': iforest_auroc})
        return SN, SA

    # Computes ROC as well as plotting histogram of the data 
    def iforest_score(self,features_train,features_test, features_ood):    
        SN, SA = self.iforest_score_basic(features_train, features_test, features_ood)

        # histogram of the confidence scores for the true data
        true_data = [[s] for s in SN]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({'True_data_iForest_Anomaly_scores': wandb.plot.histogram(true_table, "scores",title="iForest_true_data_Anomaly_scores")})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in SA]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({'OOD_data_iForest_Anomaly_scores': wandb.plot.histogram(ood_table, "scores",title="iForest_OOD_data_Anomaly_scores")})

    def get_iforest_auroc_basic(self,trainer,pl_module):
        features_train,features_test,features_ood = self.get_all_features(trainer,pl_module)
        self.iforest_score_basic(features_train,features_test,features_ood)
    
    def get_iforest_auroc(self,trainer,pl_module):
        features_train,features_test,features_ood = self.get_all_features(trainer,pl_module)
        self.iforest_score(features_train,features_test,features_ood)
