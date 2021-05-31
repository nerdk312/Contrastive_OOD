import os
from pytorch_lightning import callbacks 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.run.general_hierarchy_run_setup import train_run_name, eval_run_name, Datamodule_selection, callback_dictionary, specific_callbacks
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.datamodules.datamodule_dict import dataset_dict

# Train takes in params, a particular training module as well a model_function to instantiate the model
def train(params,model_module,model_function):
    run = wandb.init(entity="nerdk312",config = params, project= params['project'], reinit=True,group=params['group'], notes=params['notes'])  # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    #wandb.run.name = run_name(config)
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(dataset_dict,config['dataset'],config)
    #OOD_datamodule = Datamodule_selection(dataset_dict,config['OOD_dataset'],config)
    #channels = Channel_selection(dataset_dict,config['dataset'])

    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    #import ipdb; ipdb.set_trace()
    callback_dict = callback_dictionary(datamodule, config)
    desired_callbacks = specific_callbacks(callback_dict, config['callbacks'])
    #callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    
    '''
    desired_callbacks = [callback_dict['Metrics_instance_fine'],callback_dict['Metrics_fine_fine'],callback_dict['Metrics_coarse_coarse'],
                        callback_dict['Mahalanobis_instance_fine'],callback_dict['Mahalanobis_fine_fine'],callback_dict['Mahalanobis_coarse_coarse'],
                        callback_dict['Visualisation_instance_fine'],callback_dict['Visualisation_fine_fine'],callback_dict['Visualisation_coarse_coarse'],
                        callback_dict['MMD_instance'],callback_dict['Model_saving']]
    '''
    #desired_callbacks = [callback_dict['Aggregated_Mahalanobis']]
                        
    # model_function takes in the model module and the config and uses it to instantiate the model
    model = model_function(model_module,config,datamodule)


    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)#,auto_lr_find = True)
    '''
    trainer.tune(model)
    # Updates new learning rate from the learning rate finder for the saving of the config as well as the run name
    wandb.config.update({"learning_rate": model.hparams.learning_rate},allow_val_change=True)
    '''
    wandb.run.name = train_run_name(model.name,config)

    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
    
    run.finish()