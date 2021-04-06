import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score


class circular_visualisation(pl.Callback):
    def __init__(self, Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        #import ipdb; ipdb.set_trace()
        # Obtain class names list for the case of the indomain data

    def on_validation_epoch_end(self, trainer, pl_module):
        collated_features = []
        collated_labels = []
        dataloader = self.Datamodule.train_dataloader()
        for data, labels, indices in dataloader:
            if isinstance(data, tuple) or isinstance(data, list):
                data, *aug_data = data
            data = data.to(pl_module.device)
            feature_vector = pl_module.feature_vector(data)
            collated_features.append(feature_vector)

            collated_labels.append(labels)
            

        collated_features = torch.cat(collated_features)
        collated_features = collated_features.cpu().numpy()

        collated_labels = torch.cat(collated_labels)
        collated_labels = collated_labels.numpy()

        #import ipdb; ipdb.set_trace()

        theta = np.radians(np.linspace(0,360,300))
        x_2 = np.cos(theta)
        y_2 = np.sin(theta)


        plt.plot(x_2, y_2, '--', color='gray', label='Unit Circle')
        for i in range(4):
            loc = np.where(collated_labels==i)[0] # Get all the datapoints of a specific class
            plt.scatter(collated_features[loc,0], collated_features[loc,1])#, color=list(colors[loc,:]), s=60) # plot the data points for a specific class
            plt.savefig('unitnorm.png')
        #ax[1].scatter(x_base_t1[:,0], x_base_t1[:,1], marker='x', color='r', s=60) # Plot the data for the embedding of the x test line
        #ax[1].scatter(x_base_t2[:,0], x_base_t2[:,1], marker='x', color='black', s=60)
        #ax[1].scatter(y_base_t1[:,0], y_base_t1[:,1], marker='^', color='brown', s=60)
        #ax[1].scatter(y_base_t2[:,0], y_base_t2[:,1], marker='^', color='magenta', s=60)
        #ax[1].set_xlim([np.min(base_embed[:,0])*0.85,np.max(base_embed[:,0]*1.15)]) # Set an x limit for the graph
        #ax[1].set_ylim([np.min(base_embed[:,1])*1.15,np.max(base_embed[:,1]*0.85)]) # Set y limit for the graph

        


        '''    
        # Plotting the second image
        ax[1].plot(x_2, y_2, '--', color='gray', label='Unit Circle')
        for i in range(len(lines)):
            loc = np.where(labels==i)[0] # Get all the datapoints of a specific class
            ax[1].scatter(base_embed[loc,0], base_embed[loc,1], color=list(colors[loc,:]), s=60) # plot the data points for a specific class
        ax[1].scatter(x_base_t1[:,0], x_base_t1[:,1], marker='x', color='r', s=60) # Plot the data for the embedding of the x test line
        ax[1].scatter(x_base_t2[:,0], x_base_t2[:,1], marker='x', color='black', s=60)
        ax[1].scatter(y_base_t1[:,0], y_base_t1[:,1], marker='^', color='brown', s=60)
        ax[1].scatter(y_base_t2[:,0], y_base_t2[:,1], marker='^', color='magenta', s=60)
        ax[1].set_xlim([np.min(base_embed[:,0])*0.85,np.max(base_embed[:,0]*1.15)]) # Set an x limit for the graph
        ax[1].set_ylim([np.min(base_embed[:,1])*1.15,np.max(base_embed[:,1]*0.85)]) # Set y limit for the graph
        '''

class data_visualisation(pl.Callback): 
    def __init__(self, Datamodule, OOD_Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule

        self.Datamodule.setup()
        self.OOD_Datamodule.setup()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.visualise_data()

    def visualise_data(self):
        for i in range(self.Datamodule.n_lines):
            Data_loc = np.where(self.Datamodule.train_labels ==i)[0] # gets all the indices where the label has a certain index (this is correct I believe)
            OOD_Data_loc = np.where(self.OOD_Datamodule.train_labels ==i)[0]
            plt.scatter(self.Datamodule.train_data[Data_loc, 0], self.Datamodule.train_data[Data_loc, 1])
            plt.scatter(self.OOD_Datamodule.train_data[OOD_Data_loc, 0], self.OOD_Datamodule.train_data[OOD_Data_loc, 1])

        plt.savefig('full_visual.png')
        plt.savefig('full_visual.pdf')
        plt.close()
    

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

        return scores, accuracies,outputs

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
        '''
        second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        wandb.log({"AUROC_v2":second_check_roc_auc})
        '''

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

        '''
        # Double checking the calculation of the ROC scores for the data
        #second_check_scores = #1 + scores # scores are all negative values
        second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        wandb.log({"AUROC_v2":second_check_roc_auc})
        '''

        accuracy = np.mean(accuracies[: len(self.Datamodule.test_dataset)]) # Nawid - obtainn accuracy for the true dataset
        roc_auc = roc_auc_score(anomaly_targets, scores) # Nawid - obtain roc auc for a binary classification problem where the scores show how close it is to a centroid (which will say whether it is in domain or out of domain), and whether it identified as being in domain or out of domain.

        wandb.log({"AUROC":roc_auc})

        return accuracy, roc_auc


        





    



    

