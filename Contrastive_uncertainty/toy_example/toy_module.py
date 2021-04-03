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
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,):

        super().__init__()
        self.model = model
        self.datamodule = datamodule # Used for the purpose of obtaining data loader for the case of epoch starting
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.model.loss_function(batch, self.auxillary_data)
        return loss
        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
        
        self.log('Training Instance Loss', loss.item(),on_epoch=True)
        self.log('Training Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Training Instance Accuracy @ 5',acc1.item(),on_epoch = True)
        
        return loss
        '''
        #return {'loss': loss, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        loss = self.model.loss_function(batch,self.auxillary_data)
        '''
        loss, acc1, acc5 = self.model.loss_function(batch)
            
        self.log('Validation Instance Loss', loss.item(),on_epoch=True)
        self.log('Validation Instance Accuracy @ 1',acc1.item(),on_epoch = True)
        self.log('Validation Instance Accuracy @ 5',acc1.item(),on_epoch = True)        
        '''
    def test_step(self, batch, batch_idx):
        loss = self.model.loss_function(batch,self.auxillary_data)
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
    
    def on_train_epoch_start(self, epoch):

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

        #  (The only part that I added) - Used to call the cluster results for the case of the prototypical 
        self.auxillary_data = model.on_train_epoch_start(self.datamodule)
        



#encoder = MocoToy()
#encoder = SupConToy()
#Model = Toy(encoder)
