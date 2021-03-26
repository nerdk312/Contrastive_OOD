import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from datamodules.cifar10_datamodule import CIFAR10DataModule
from datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from datamodules.mnist_datamodule import MNISTDataModule
from datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms

from metrics.metric_callback import MetricLogger,evaluation_metrics,evaltypes

from Moco.moco_run_setup import SimCLR_run_name, Datamodule_selection,Channel_selection,callback_dictionary
from SIMCLR.simclr_module import SimCLR


def train(params):
    wandb.init(entity="nerdk312",config = params,project= params['project']) # Required to have access to wandb config, which is needed to set up a sweep
    wandb_logger = WandbLogger(log_model=True,sync_step=False,commit=False)
    config = wandb.config

    # Run setup
    wandb.run.name = SimCLR_run_name(config)
    pl.seed_everything(config['seed'])

    datamodule = Datamodule_selection(config['dataset'],config)
    OOD_datamodule = Datamodule_selection(config['OOD_dataset'],config)
    channels = Channel_selection(config['dataset'])

    model = SimCLR(batch_size = config['bsz'],num_samples = config['num_samples'],
    num_channels = channels,warmup_epochs = config['warmup_epochs'],
    lr = config['lr'],opt_weight_decay = config['opt_weight_decay'],
    loss_temperature = config['loss_temperature'])

    
    wandb_logger.watch(model, log='gradients', log_freq=100) # logs the gradients

    trainer = pl.Trainer(fast_dev_run = config['fast_run'],progress_bar_refresh_rate=20,
                        limit_train_batches = config['training_ratio'],limit_val_batches=config['validation_ratio'],limit_test_batches = config['test_ratio'],
                        max_epochs = config['epochs'],check_val_every_n_epoch = config['val_check'],
                        gpus=1,logger=wandb_logger,checkpoint_callback = False,deterministic =True)#,callbacks = desired_callbacks)

    trainer.fit(model,datamodule)
    trainer.test(datamodule=datamodule,
            ckpt_path=None)  # uses last-saved model , use test set to call the reliability diagram only at the end of the training process