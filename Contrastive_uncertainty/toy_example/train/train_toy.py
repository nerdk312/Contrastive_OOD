import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


#from Contrastive_uncertainty.toy_example.callbacks.toy_visualisation_callbacks import circular_visualisation, data_visualisation
from Contrastive_uncertainty.toy_example.run.toy_run_setup  import callback_dictionary, Datamodule_selection, Model_selection

from Contrastive_uncertainty.toy_example.models.toy_moco import MocoToy
from Contrastive_uncertainty.toy_example.models.toy_softmax import SoftmaxToy
from Contrastive_uncertainty.toy_example.models.toy_PCL import PCLToy
from Contrastive_uncertainty.toy_example.models.toy_supcon import SupConToy
from Contrastive_uncertainty.toy_example.models.toy_ova import OVAToy


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
    datamodule.setup(), OOD_datamodule.setup()

    # Model for the task
    #encoder = MocoToy(config['hidden_dim'],config['embed_dim'])
    #encoder = SoftmaxToy(config['hidden_dim'],config['embed_dim'])
    #model = PCLToy(datamodule= datamodule)
    #model = Toy(encoder, datamodule=datamodule)


    #model = SoftmaxToy(datamodule = datamodule)
    #model = OVAToy(datamodule=datamodule)
    model = Model_selection(datamodule, config)
    '''    
    model = MocoToy(datamodule=datamodule,
                    optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                    momentum=config['momentum'], weight_decay=config['weight_decay'],
                    hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                    num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                    softmax_temperature=config['softmax_temperature'],
                    pretrained_network=config['pretrained_network'], num_classes=config['num_classes'])
    '''
    
    '''
    model = SupConToy(datamodule=datamodule,
                    optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                    momentum=config['momentum'], weight_decay=config['weight_decay'],
                    hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                    softmax_temperature=config['softmax_temperature'],base_temperature=config['softmax_temperature'],
                    num_classes= config['num_classes'])
    '''
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    desired_callbacks = [callback_dict['Uncertainty_visualise']]#[callback_dict['ROC'],callback_dict['Mahalanobis']]

    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients
        
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)
    
    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process
