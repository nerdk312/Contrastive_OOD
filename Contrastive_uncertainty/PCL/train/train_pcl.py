import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.PCL.run.pcl_run_setup import run_name, Datamodule_selection,Channel_selection,callback_dictionary
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule


def training(params):
    wandb.init(entity="nerdk312",config = params,project= params['project']) # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(config['dataset'],config)
    OOD_datamodule = Datamodule_selection(config['OOD_dataset'],config)
    channels = Channel_selection(config['dataset'])
    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    
    
    desired_callbacks = [callback_dict['Confusion_matrix'],callback_dict['ROC'],
                        callback_dict['Reliability'],callback_dict['Metrics'], callback_dict['Model_saving'], 
                        callback_dict['Mahalanobis'], callback_dict['Mahalanobis_compressed'],callback_dict['Euclidean'],
                        callback_dict['MMD'],callback_dict['Visualisation'],callback_dict['Centroid'],callback_dict['Uniformity'],
                        callback_dict['SupCon']]
    
    
    #desired_callbacks = []

    model = PCLModule(datamodule=datamodule,optimizer=config['optimizer'],
    learning_rate=config['learning_rate'],momentum=config['momentum'],
    weight_decay=config['weight_decay'],num_classes=config['num_classes'],
    class_dict=class_names_dict, emb_dim=config['emb_dim'],
    num_negatives=config['num_negatives'],
    encoder_momentum=config['encoder_momentum'],softmax_temperature=config['softmax_temperature'],
    num_cluster=config['num_cluster'], use_mlp=config['use_mlp'],
    num_channels=channels, classifier=config['classifier'],
    normalize=config['normalize'], instance_encoder=config['instance_encoder'],
    pretrained_network=config['pretrained_network'])
        

    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)#,auto_lr_find = True)
    
    wandb.run.name = run_name(config)
        
    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process