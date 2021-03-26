import os 
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms

from Contrastive_uncertainty.metrics.metric_callback import MetricLogger,evaluation_metrics,evaltypes

from Contrastive_uncertainty.Moco.self_supervised import SSLOnlineEvaluator
from Contrastive_uncertainty.Moco.ssl_finetuner import SSLFineTuner
from Contrastive_uncertainty.Moco.moco_callbacks import ModelSaving,OOD_ROC, OOD_confusion_matrix,ReliabiltyLogger,ImagePredictionLogger
from Contrastive_uncertainty.Moco.moco_run_setup import run_name, Datamodule_selection,Channel_selection,callback_dictionary
from Contrastive_uncertainty.Moco.moco2_module import MocoV2

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
    class_names_dict = datamodule.idx2class # name of dict which contains class names
    callback_dict = callback_dictionary(datamodule,OOD_datamodule,config)
    desired_callbacks = [callback_dict['Confusion_matrix'],callback_dict['ROC'],
                        callback_dict['Reliability'],callback_dict['Metrics'],callback_dict['Model_saving'],callback_dict['Mahalanobis']]


    model = MocoV2(emb_dim = config['emb_dim'],num_negatives = config['num_negatives'],
        encoder_momentum = config['encoder_momentum'], softmax_temperature = config['softmax_temperature'],
        optimizer = config['optimizer'],learning_rate = config['learning_rate'],
        momentum = config['momentum'], weight_decay = config['weight_decay'],
        batch_size = config['bsz'],use_mlp = config['use_mlp'], z_dim = config['z_dim'],
        num_classes = config['num_classes'],datamodule = datamodule,num_channels = channels,
        classifier = config['classifier'],normalize = config['normalize'],contrastive = config['contrastive'],
        class_dict = class_names_dict,instance_encoder = config['instance_encoder'],pretrained_network = config['pretrained_network'])


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


