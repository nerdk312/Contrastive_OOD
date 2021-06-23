import os
from torch._C import import_ir_module 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.toy_replica.toy_general.run.general_run_setup import train_run_name, eval_run_name,Datamodule_selection,callback_dictionary, specific_callbacks 
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import dataset_dict, OOD_dict
from Contrastive_uncertainty.toy_replica.toy_general.utils.hybrid_utils import previous_model_directory
from Contrastive_uncertainty.toy_replica.cross_entropy.models.cross_entropy_module import CrossEntropyToy

def evaluation(run_path, update_dict, model_module, model_function):
    api = wandb.Api()
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    run = wandb.init(id=previous_run.id,resume='allow',project=previous_config['project'],group=previous_config['group'], notes=previous_config['notes'])
    
    #run = wandb.init(entity="nerdk312",config = params, project= params['project'], reinit=True,group=params['group'], notes=params['notes'])  # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True, sync_step=False, commit=False)
    config = previous_config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)

    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(dataset_dict, config['dataset'],config)
    
    # CHANGE SECTION
    # Load from checkpoint using pytorch lightning loads everything directly to continue training from the class function
    # model = model_module.load_from_checkpoint(model_dir)
    model = model_function(model_module, config, datamodule)

    # Obtain checkpoint for the model        
    model_dir = 'Models'
    model_dir = previous_model_directory(model_dir, run_path) # Used to preload the model

    # Updates OOD dataset if not manually specified in the update dict
    if 'OOD_dataset' in update_dict:
        pass
    else: #  Cannot update the Update dict directly as this will carry on for other simulations
        config['OOD_dataset'] = OOD_dict[config['dataset']]
        #update_dict['OOD_dataset'] = OOD_dict[config['dataset']]
        #print('updated dict')

    # Update the trainer and the callbacks for a specific test
    
    '''
    for update_k, update_v in update_dict.items():
        if update_k in config:
            config[update_k] = update_v
    '''
    new_config_params = ['callbacks','typicality_bootstrap','typicality_batch']

    for update_k, update_v in update_dict.items():
        '''
        if update_k =='callbacks':
            config[update_k] = update_v
        '''
        if update_k in new_config_params:
            config[update_k] = update_v
             
        if update_k in config:
            if update_k =='epochs':
                config[update_k] = config[update_k] + update_v
            else:
                config[update_k] = update_v

    callback_dict = callback_dictionary(datamodule, config)
    desired_callbacks = specific_callbacks(callback_dict, config['callbacks'])
    #wandb.config.update(config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)        
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks,
                        resume_from_checkpoint=model_dir)#,auto_lr_find = True)
    trainer.fit(model)
    
    trainer.test(model,datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
     
    run.finish()