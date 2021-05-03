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


def resume(run_path):
    api = wandb.Api()
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    # Obtain the previous logged value
    history = previous_run.history()
    previous_epoch = history.epoch.iloc[-1]
    #import ipdb; ipdb.set_trace()

    run = wandb.init(id=previous_run.id,resume='allow',project=previous_config['project'])
    
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = previous_config
    #import ipdb; ipdb.set_trace()
    # Change None string to None type
    '''
    if config['pretrained_network'] == "None":
        config['pretrained_network'] = None

    model_dir = 'Toy_Models'
    model_dir = os.path.join(model_dir,run_path)
    for i in os.listdir(model_dir):
        if 'Test' in i:
            test_model = i
    model_dir = os.path.join(model_dir,test_model)
    config['pretrained_network'] = model_dir
    '''
    #import ipdb; ipdb.set_trace()

    
    folder = 'Images'
    if not os.path.exists(folder):
        os.mkdir(folder)
    #import ipdb; ipdb.set_trace()
    # Run setup
    
    pl.seed_everything(config['seed'])
    
    datamodule = Datamodule_selection(config['dataset'],config)
    OOD_datamodule = Datamodule_selection(config['OOD_dataset'],config)
    datamodule.setup(), OOD_datamodule.setup()

    
    model = Model_selection(datamodule, config)
    callback_dict = callback_dictionary(datamodule, OOD_datamodule, config)
    #desired_callbacks = [callback_dict['Uncertainty_visualise']]#[callback_dict['ROC'],callback_dict['Mahalanobis']]
    desired_callbacks = [callback_dict['Saving']]
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    total_epochs = previous_epoch + config['epochs']    
    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = total_epochs, check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True,callbacks = desired_callbacks)
    trainer.current_epoch = previous_epoch

    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process