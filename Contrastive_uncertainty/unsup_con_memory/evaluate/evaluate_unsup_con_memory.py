import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.unsup_con_memory.datamodules.datamodule_dict import dataset_dict

from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule
from Contrastive_uncertainty.unsup_con_memory.run.unsup_con_memory_run_setup import \
    train_run_name,eval_run_name, Datamodule_selection, Channel_selection, callback_dictionary


def evaluation(run_path):
    api = wandb.Api()
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    run = wandb.init(id=previous_run.id,resume='allow',project=previous_config['project'],group=previous_config['group'], notes=previous_config['notes'])
    
    #run = wandb.init(entity="nerdk312",config = params, project= params['project'], reinit=True,group=params['group'], notes=params['notes'])  # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = previous_config
    # Change None string to None type
    if config['pretrained_network'] == "None":
        config['pretrained_network'] = None

    model_dir = 'Models'
    model_dir = os.path.join(model_dir,run_path)
    for i in os.listdir(model_dir):
        if 'Test' in i:
            test_model = i
    model_dir = os.path.join(model_dir,test_model)
    config['pretrained_network'] = model_dir
    #import ipdb; ipdb.set_trace()


    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
    #import ipdb; ipdb.set_trace()
    # Run setup
    
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(dataset_dict,config['dataset'],config)
    OOD_datamodule = Datamodule_selection(dataset_dict,config['OOD_dataset'],config)
    channels = Channel_selection(dataset_dict,config['dataset'])

    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    
    desired_callbacks = [callback_dict['Metrics'], callback_dict['Model_saving'], 
                        callback_dict['Mahalanobis'],callback_dict['MMD'],callback_dict['Visualisation'],callback_dict['Uniformity']] 
    
    #desired_callbacks = []

    # Hack to be able to use the num clusters with wandb sweep since wandb sweep cannot use a list of lists I believe
    if isinstance(config['num_cluster'], list) or isinstance(config['num_cluster'], tuple):
        num_clusters = config['num_cluster']
    else:  
        num_clusters = [config['num_cluster']]

    model = UnSupConMemoryModule(emb_dim = config['emb_dim'],num_negatives = config['num_negatives'],
        memory_momentum = config['memory_momentum'],num_cluster=num_clusters, 
        softmax_temperature = config['softmax_temperature'],
        optimizer = config['optimizer'],learning_rate = config['learning_rate'],
        momentum = config['momentum'], weight_decay = config['weight_decay'],
        use_mlp = config['use_mlp'],
        datamodule = datamodule,num_channels = channels,
        instance_encoder = config['instance_encoder'],
        pretrained_network = config['pretrained_network'])
        
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)#,auto_lr_find = True)
   
    
    trainer.test(model,datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
    
    
    run.finish()
