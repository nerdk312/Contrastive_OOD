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

from Contrastive_uncertainty.general_clustering.general_callbacks import ModelSaving, MMD_distance, \
                                                                           Uniformity, SupConLoss, Centroid_distance, \
                                                                           quickloading

# Override method from the general section
class ModelSaving(pl.Callback):
    def __init__(self,interval):
        super().__init__()
        self.interval = interval
        self.counter = interval
        #self.epoch_last_check = 0
    # save the state dict in the local directory as well as in wandb
        
    def on_test_epoch_end(self, trainer, pl_module): # save during the test stage
        epoch =  trainer.current_epoch
        self.save_model(pl_module,epoch)

    def save_model(self,pl_module,epoch):
        filename = f"CurrentEpoch:{epoch}_" + wandb.run.name + '.pt' 
        print('filename:',filename)
        torch.save({
            'encoder_state_dict':pl_module.encoder.state_dict(),
        },filename)
        wandb.save(filename)