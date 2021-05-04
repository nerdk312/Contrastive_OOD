import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean


class Toy(pl.LightningModule):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,):

        super().__init__()
        self.datamodule = datamodule # Used for the purpose of obtaining data loader for the case of epoch starting
        #self.save_hyperparameters()

        self.auxillary_data = None #self.aux_data() #self.on_train_epoch_start(self.datamodule)

    def training_step(self, batch, batch_idx):
        self._momentum_update_key_encoder()

        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
        #return {'loss': loss, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        
        #import ipdb; ipdb.set_trace()
        for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)

        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
        
        self.log('Test Instance Loss', loss.item(),on_epoch=True)
        self.log('Test Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Test Instance Accuracy @ 5',acc1.item(),on_epoch = True)
        '''

    def loss_function(self, batch, auxillary_data=None):
        raise NotImplementedError

    def _momentum_update_key_encoder(self):
        raise NotImplementedError
    

    def aux_data(self):
        ''' Placeholder function, used to obtain auxillary data for a specific task'''
        auxillary_data = None
        return auxillary_data

    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        # Save model at this location
        return optimizer
    