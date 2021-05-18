import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.general_clustering.run.general_clustering_run_setup import train_run_name, eval_run_name,Datamodule_selection,callback_dictionary
from Contrastive_uncertainty.general_clustering.datamodules.datamodule_dict import dataset_dict


# Train takes in params, a particular training module as well a model_function to instantiate the model
def train(params,model_module,model_function):
    run = wandb.init(entity="nerdk312", config = params, project=params['project'], reinit=True, group=params['group'], notes=params['notes'])  # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    #wandb.run.name = run_name(config)
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(dataset_dict,config['dataset'],config)
    OOD_datamodule = Datamodule_selection(dataset_dict,config['OOD_dataset'],config)
    #channels = Channel_selection(dataset_dict,config['dataset'])

    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    
    desired_callbacks = [callback_dict['Metrics'], callback_dict['Model_saving'], 
                        callback_dict['Mahalanobis'],callback_dict['MMD'],callback_dict['Visualisation'],callback_dict['Uniformity']] 
    
    # desired_callbacks = []
    # model_function takes in the model module and the config and uses it to instantiate the model
    
        
    # Need to add the num clusters argument also for this case
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
    
    '''
    # finetune
    print('fine tuning')
    tuner = SSLFineTuner(model, in_features=model.z_dim, num_classes=model.num_classes)
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,max_epochs = config['epochs'],
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(tuner,datamodule)
    '''
    run.finish()
