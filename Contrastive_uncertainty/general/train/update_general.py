import enum
import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger



def update_config(run_path, update_dict):
    api = wandb.Api()
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config

    config = previous_config

    for update_k, update_v in update_dict.items():
        if update_k == 'group' or update_k =='notes':
            config[update_k] = update_v

    # During run, this is when the group, notes and config are able to change for the task
    run = wandb.init(id=previous_run.id,resume='allow',project=config['project'],group=config['group'], notes=config['notes'])
    wandb.config.update(config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
    run.finish()
