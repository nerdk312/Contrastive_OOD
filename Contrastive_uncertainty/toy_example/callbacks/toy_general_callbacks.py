import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

class ModelSaving(pl.Callback):
    def __init__(self,interval):
        super().__init__()
        self.interval = interval
        self.counter = interval
        #self.epoch_last_check = 0
    # save the state dict in the local directory as well as in wandb
    
    def on_validation_epoch_end(self,trainer,pl_module): # save every interval
        epoch = trainer.current_epoch
        print('Epoch:',epoch)
        #import ipdb;ipdb.set_trace()
        #  Checks if it is equal to or more than the value
        if epoch >= self.counter:
            self.save_model(trainer, pl_module, epoch)
            # Makes it equal to the value adn then increases further
            self.counter = epoch +self.interval # Increase the interval
                
    
    def on_test_epoch_end(self, trainer, pl_module): # save during the test stage
        epoch =  trainer.current_epoch
        self.save_model(trainer, pl_module, epoch)

    
    def save_model(self, trainer, pl_module, epoch):
        folder = 'Toy_Models'
        folder = os.path.join(folder, wandb.run.path)
        # makedirs used to make multiple subfolders in comparison to mkdir which makes a single folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Choose different name based on whether on test stage or validation stage
        filename = f'TestModel:{epoch}.pt' if trainer.testing else f'Model:{epoch}.pt'
        filename = os.path.join(folder,filename)
        #import ipdb; ipdb.set_trace()

        torch.save({
            'optimizer_state_dict':pl_module.optimizers().state_dict(),
            'encoder_state_dict':pl_module.encoder.state_dict(),
            'classifier_state_dict':pl_module.classifier.state_dict(),
        },filename)