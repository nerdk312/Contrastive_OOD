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

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix


# Used to save the model in the directory as well as in wandb
class ModelSaving(pl.Callback):
    def __init__(self, interval,folder_name):
        super().__init__()
        self.interval = interval
        self.counter = interval
        self.folder_name = folder_name
        #self.epoch_last_check = 0
    
    
    def on_validation_epoch_end(self,trainer,pl_module): # save every interval
        epoch = trainer.current_epoch 
        # Check if it is equal to or more than the value
        if epoch >= self.counter:
            self.save_model(trainer, pl_module, epoch)
            self.counter = epoch + self.interval # Increase the interval
    
    def on_test_epoch_end(self, trainer, pl_module):  # save during the test stage
        epoch = trainer.current_epoch
        self.save_model(trainer,pl_module, epoch)

        
    def save_model(self, trainer, pl_module,epoch):
        folder = self.folder_name
        folder = os.path.join(folder, wandb.run.path)
        # makedirs used to make multiple subfolders in comparison to mkdir which makes a single folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Choose different name based on whether on test stage or validation stage
        filename = f'TestModel:{epoch}.ckpt' if trainer.testing else f'Model:{epoch}.ckpt'
        filename = os.path.join(folder,filename)

        # Saves the checkpoint to enable to continue loading
        trainer.save_checkpoint(filename)
       
                
# Calculation of MMD based on the definition 2/ equation 1 in the paper        
# http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchetal12.pdf?origin=publication_detail
class MMD_distance(pl.Callback):
    def __init__(self,Datamodule,
        vector_level:str ='instance',
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback
        self.log_name = 'MMD_distance'

        self.vector_level = vector_level

    def on_test_epoch_end(self,trainer, pl_module):
        self.calculate_MMD(pl_module)
    
    # Log MMD whilst the network is training
    def on_validation_epoch_end(self, trainer,pl_module):
        self.calculate_MMD(pl_module)

    def calculate_MMD(self, pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector}}
        dataloader = self.Datamodule.train_dataloader()
        with torch.no_grad():
            MMD_values = []
            low = torch.tensor(-1.0).to(device=pl_module.device)
            high = torch.tensor(1.0).to(device=pl_module.device)
            uniform_distribution = torch.distributions.uniform.Uniform(low,high) # causes all samples to be on the correct device when obtainig smaples https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
            #uniform_distribution =  torch.distributions.uniform.Uniform(-1,1).sample(output.shape)
            loader = quickloading(self.quick_callback, dataloader) # Used to get a single batch or used to get the entire dataset
            assert len(loader)>0, 'loader is empty'

            for img, *label,indices in loader:
                if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

                img = img.to(pl_module.device)
                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = self.vector_dict['vector_level'][self.vector_level](img) # Performs the callback for the desired level 
                
                uniform_samples = uniform_distribution.sample(output.shape)
                uniform_samples = torch.nn.functional.normalize(uniform_samples,dim=1) # obtain normalized representaitons on a hypersphere
                # calculate the difference between the representation and samples from a unifrom distribution on a hypersphere
                diff = output - uniform_samples
                MMD_values.append(diff.cpu().numpy())

        MMD_list = np.concatenate(MMD_values)
        MMD_dist = np.mean(MMD_list)
        # Logs the MMD distance into wandb
        wandb.log({f'{self.log_name}:{self.vector_level}':MMD_dist})

        return MMD_dist

# choose whether to iterate over the entire dataset or over a single batch of the data (https://github.com/pytorch/pytorch/issues/1917 - obtaining a single batch)
def quickloading(quick_test, dataloader):
    if quick_test:
        loader = [next(iter(dataloader))] # This obtains a single batch of the data as a list which can be iterated
    else:
        loader = dataloader
    return loader


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

def get_roc_plot(xin, xood,OOD_name):
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
    
    ROC_filename = f'Images/ROC_{OOD_name}.png'
    plt.savefig(ROC_filename)
    wandb_ROC = f'ROC curve: OOD dataset {OOD_name}'
    wandb.log({wandb_ROC:wandb.Image(ROC_filename)})
