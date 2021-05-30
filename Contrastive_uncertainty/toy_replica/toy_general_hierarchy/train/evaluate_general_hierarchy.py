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

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.run.general_hierarchy_run_setup import train_run_name, eval_run_name,Datamodule_selection, callback_dictionary, specific_callbacks
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.datamodules.datamodule_dict import dataset_dict 
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.utils.hybrid_utils import previous_model_directory
script_params_dict = {'test_experiments': [1,2],
                    'OOD_dataset': ['TwoMoons',
                                    'Diagonal']
}
#import ipdb; ipdb.set_trace()

def evaluation(run_path,model_module, model_function):
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
    # Make it so that the datamodule and the OOD datamodule can be obtained using one function
    datamodule = Datamodule_selection(dataset_dict,config['dataset'],config)
    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    model = model_function(model_module, config, datamodule)
    
    # CHANGE SECTION
    # Load from checkpoint using pytorch lightning loads everything directly to continue training from the class function
    # model = model_module.load_from_checkpoint(model_dir)
    # Obtain checkpoint for the model        
    model_dir = 'Models'
    model_dir = previous_model_directory(model_dir, run_path) # Used to preload the model

    #wandb.config.update(config, allow_val_change=True) # Updates the config (particularly used to increase the number of epochs present)
    
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients
    for i in range(len(script_params_dict['test_experiments'])):
        for script_key, script_value in script_params_dict.items():
            # Update the params
            if script_key in config:
                config[script_key] = script_value[i]

        #OOD_datamodule = Datamodule_selection(dataset_dict, 'Diagonal', config)
        OOD_datamodule = Datamodule_selection(dataset_dict, config['OOD_dataset'], config)

        callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
        names = ['Mahalanobis_instance_fine','Mahalanobis_fine_fine','Mahalanobis_coarse_coarse']
        desired_callbacks = specific_callbacks(callback_dict, names)
        
        
        '''
        desired_callbacks = [callback_dict['Metrics_instance_fine'],callback_dict['Metrics_fine_fine'],callback_dict['Metrics_coarse_coarse'],
                            callback_dict['Mahalanobis_instance_fine'],callback_dict['Mahalanobis_fine_fine'],callback_dict['Mahalanobis_coarse_coarse'],
                            callback_dict['Visualisation_instance_fine'],callback_dict['Visualisation_fine_fine'],callback_dict['Visualisation_coarse_coarse'],
                            callback_dict['MMD_instance'],callback_dict['Model_saving']]
        '''
        #desired_callbacks = []

        trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                            limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                            max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                            gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks,
                            resume_from_checkpoint=model_dir) #,auto_lr_find = True)

        trainer.test(model,datamodule=datamodule,
                ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process

    run.finish()