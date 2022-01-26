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
import statistics

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn, get_roc_plot, table_saving
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k





# Performs the typicality test between the ID test data and the OOD data
class ConfusionLogProbability(pl.Callback):
    def __init__(self, Confusion_Datamodule,
        quick_callback:bool = True):
        
        super().__init__()

        self.Confusion_Datamodule = Confusion_Datamodule
        self.quick_callback = quick_callback


    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module)

    def forward_callback(self, trainer, pl_module):
        ## Initialisation##
        self.Confusion_Datamodule.setup()
        self.ID_dataname = self.Confusion_Datamodule.ID_Datamodule.name
        self.OOD_dataname = self.Confusion_Datamodule.OOD_Datamodule.name

        ood_test_loader = self.Confusion_Datamodule.ood_dataloader()

        test_predictions = self.get_predictions(pl_module, ood_test_loader)
        self.confusion_calculation(test_predictions)
        


    # Obtain the predictions for all the different modules
    def get_predictions(self, pl_module, dataloader):
        
        loader = quickloading(self.quick_callback, dataloader)
        all_predictions = []
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label = label[0]
                

            img = img.to(pl_module.device) 
            model_predictions = []
            # Go through each member in the ensemble and obtain a classification prediction
            for i in range(pl_module.num_models):
                logits = pl_module.class_forward(img, i)
                predictions = F.softmax(logits,dim=1)
                

                model_predictions.append(predictions)
            # Join the predictions from the models togehter for a particular back of data points    
            model_predictions = torch.stack(model_predictions) # shape (Num models, batch, num classes)
            all_predictions.append(model_predictions)

        # concatenate all the batches to get all the predictions
        all_predictions = torch.cat(all_predictions, dim=1)
        return all_predictions
                
    
    def confusion_calculation(self, predictions):
        '''
        Args:
            Predictions: shape  (num models, datasize, total classes)
        '''
        inlier_clases = self.Confusion_Datamodule.ID_Datamodule.num_classes

        predictions = torch.mean(predictions, dim=0) # shape (datasize, total classes)
        inlier_predictions = predictions[:,0:inlier_clases] # Predictions for inlier classes only shape (datasize, num_inlier_class)
        
        classwise_CLP = torch.log(torch.mean(inlier_predictions,dim=0))    # calculate mean of data samples and calculate he log
        #
        #import ipdb; ipdb.set_trace()
        ## Perfrom summation over all inlier classes, then calculate the mean and the log
        

        CLP = torch.sum(inlier_predictions, dim=1)
        CLP = torch.log(torch.mean(CLP))
        min_classwise_CLP = torch.min(classwise_CLP)
        max_classwise_CLP = torch.max(classwise_CLP)
        
        wandb.run.summary['Confusion Log Probability'] = CLP
        # Need to change the tensor to cpu first to log the tensor value
        wandb.run.summary['Class Wise CLP'] = classwise_CLP.cpu()
        wandb.run.summary['Min Class Wise CLP'] = min_classwise_CLP
        wandb.run.summary['Max Class Wise CLP'] = max_classwise_CLP

        # Used to make wandb table to capture all the different values present
        class_wise_clp_np= np.expand_dims(classwise_CLP.cpu().numpy(),axis=1)
        columns = ['Class Wise Confusion Log Probability']
        indices = [f'Class {i}' for i in range(inlier_clases)] 
        df = pd.DataFrame(class_wise_clp_np, columns= columns, index=indices)

        table = wandb.Table(data= df)
        wandb.log({'Class Wise Confusion Log Probability': table})

        

        '''
        CLP = torch.mean(inlier_predictions)
        '''
        

