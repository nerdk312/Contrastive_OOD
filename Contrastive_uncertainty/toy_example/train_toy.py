import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.toy_example.diagonal_lines_datamodule import DiagonalLinesDataModule
from Contrastive_uncertainty.toy_example.straight_lines_datamodule import StraightLinesDataModule
from Contrastive_uncertainty.toy_example.toy_transforms import ToyTrainDiagonalLinesTransforms, ToyEvalDiagonalLinesTransforms
from Contrastive_uncertainty.toy_example.toy_callbacks import circular_visualisation, data_visualisation
from Contrastive_uncertainty.toy_example.toy_run_setup import callback_dictionary

from Contrastive_uncertainty.toy_example.toy_moco import MocoToy
from Contrastive_uncertainty.toy_example.toy_softmax import SoftmaxToy
from Contrastive_uncertainty.toy_example.toy_PCL import PCLToy
from Contrastive_uncertainty.toy_example.toy_supcon import SupConToy




def train(params):
    wandb.init(entity="nerdk312",config = params,project= params['project']) # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    pl.seed_everything(config['seed'])

    datamodule = DiagonalLinesDataModule(32,0.1,train_transforms=ToyTrainDiagonalLinesTransforms(),test_transforms=ToyEvalDiagonalLinesTransforms())
    datamodule.setup()

    OOD_datamodule = StraightLinesDataModule(32,0.1,train_transforms=ToyTrainDiagonalLinesTransforms(),test_transforms=ToyEvalDiagonalLinesTransforms())
    OOD_datamodule.setup()

    # Model for the task
    #encoder = MocoToy(config['hidden_dim'],config['embed_dim'])
    #encoder = SoftmaxToy(config['hidden_dim'],config['embed_dim'])
    #encoder = PCLToy()
    #model = Toy(encoder, datamodule=datamodule)

    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    desired_callbacks = [callback_dict['ROC'],callback_dict['Mahalanobis']]
    model = SoftmaxToy(datamodule = datamodule)
    
    visualiser = data_visualisation(datamodule, OOD_datamodule)
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients
        
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)
    
    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
