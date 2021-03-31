import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.toy_example.toy_moco import MocoToy
from Contrastive_uncertainty.toy_example.toy_supcon import SupConToy

class Toy(pl.LightningModule):
    def __init__(self,model,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,):

        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.model.loss_function(batch)
        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
        
        self.log('Training Instance Loss', loss.item(),on_epoch=True)
        self.log('Training Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Training Instance Accuracy @ 5',acc1.item(),on_epoch = True)
        
        return loss
        '''
        #return {'loss': loss, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx,dataset_idx):
        loss = self.model.loss_function(batch)
        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
            
        self.log('Validation Instance Loss', loss.item(),on_epoch=True)
        self.log('Validation Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Validation Instance Accuracy @ 5',acc1.item(),on_epoch = True)        
        '''
    def test_step(self, batch, batch_idx):
        loss = self.model.loss_function(batch)
        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
        
        self.log('Test Instance Loss', loss.item(),on_epoch=True)
        self.log('Test Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Test Instance Accuracy @ 5',acc1.item(),on_epoch = True)
        '''

    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer


#encoder = MocoToy()
encoder = SupConToy()
Model = Toy(encoder)
