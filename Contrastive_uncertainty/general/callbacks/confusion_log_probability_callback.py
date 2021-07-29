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
            for i in range(pl_module.num_models):
                logits = pl_module.class_forward(img, i)
                predictions = F.softmax(logits,dim=1)
                

                model_predictions.append(predictions)
                
            model_predictions = torch.stack(model_predictions) # shape (Num models, batch, num classes)
            all_predictions.append(model_predictions)


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
        
        # Perfrom summation over all inlier classes, then calculate the mean and the log
        CLP = torch.sum(inlier_predictions, dim=1)
        CLP = torch.log(torch.mean(CLP))
        
        wandb.run.summary['Confusion Log Probability'] = CLP
        
        '''
        CLP = torch.mean(inlier_predictions)
        '''
        

