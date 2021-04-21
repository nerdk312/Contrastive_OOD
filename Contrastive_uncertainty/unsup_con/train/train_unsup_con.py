import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.unsup_con.models.unsup_con_module import SupConModule
from Contrastive_uncertainty.unsup_con.run.unsup_con_run_setup import run_name, Datamodule_selection,Channel_selection,callback_dictionary

def train(params):
    wandb.init(entity="nerdk312",config = params,project= params['project']) # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Run setup
    #wandb.run.name = run_name(config)
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(config['dataset'],config)
    OOD_datamodule = Datamodule_selection(config['OOD_dataset'],config)
    channels = Channel_selection(config['dataset'])
    class_names_dict = datamodule.idx2class  # name of dict which contains class names
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    
    desired_callbacks = [callback_dict['Metrics'], callback_dict['Model_saving'], 
                        callback_dict['Mahalanobis'],callback_dict['MMD'],callback_dict['Visualisation'],callback_dict['Uniformity']] 

    model = UnSupConModule(emb_dim = config['emb_dim'],contrast_mode=config['contrast_mode'],
        softmax_temperature = config['softmax_temperature'],
        optimizer = config['optimizer'],learning_rate = config['learning_rate'],
        momentum = config['momentum'], weight_decay = config['weight_decay'],
        use_mlp = config['use_mlp'],num_cluster = config['num_cluster'],
        datamodule = datamodule,num_channels = channels,
        instance_encoder = config['instance_encoder'],
        pretrained_network = config['pretrained_network'])
        


    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    '''
    online_evaluator = SSLOnlineEvaluator()
    #online_evaluator.to_device = to_device
    '''

    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)#,auto_lr_find = True)
    '''
    trainer.tune(model)
    # Updates new learning rate from the learning rate finder for the saving of the config as well as the run name
    wandb.config.update({"learning_rate": model.hparams.learning_rate},allow_val_change=True)
    '''
    wandb.run.name = run_name(config)
    '''    
    trainer.test(model,datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
    '''
    
    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
    
    # load the backbone
    #backbone = CPCV2.load_from_checkpoint(args.ckpt_path, strict=False)
    #model.encoder_loading('Epochs_1000_lr_3.000e-02_bsz_256_seed_42.pt')

    '''
    # finetune
    print('fine tuning')
    tuner = SSLFineTuner(model, in_features=model.z_dim, num_classes=model.num_classes)
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,max_epochs = config['epochs'],
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(tuner,datamodule)
    '''
    #trainer.test(datamodule=dm)


