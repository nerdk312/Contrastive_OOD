import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import math
from tqdm import tqdm
import faiss
import numpy as np
from timeit import default_timer as timer

from Contrastive_uncertainty.dpsup_con.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.dpsup_con.models.dpmeans import DPMeans

      

class DPSupConModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        distance_threshold: float = 1.0,
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodule = datamodule

        # create the encoders
        # num_classes is the output fc dimension
        
        self.encoder = self.init_encoders()
        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
            
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)
    
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = 10)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = 10)
        
        return encoder
        
    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')
        features = torch.zeros(self.datasize(dataloader), self.hparams.emb_dim, device = self.device)
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder(images)  # Nawid - obtain features for the task
            feat = nn.functional.normalize(feat, dim=1) # Obtain 12 normalised features for clustering
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()
    


    def cluster_data(self, dataloader):
        features = self.compute_features(dataloader)
        dpmeans = dp.FastDPMeans(features, self.cutoff)
        start = timer()
        loss  = dpmeans.run(verbose=10)
        end   = timer()
        print(end - start)



    def on_train_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        #return self.auxillary_data

    def on_fit_start(self):
        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        
    def aux_data(self, dataloader):
        cluster_result = self.cluster_data(dataloader)
        return cluster_result