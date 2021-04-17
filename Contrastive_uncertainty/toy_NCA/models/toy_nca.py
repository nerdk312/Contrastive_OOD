import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import math

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k

from Contrastive_uncertainty.toy_NCA.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_NCA.models.toy_module import Toy
from Contrastive_uncertainty.toy_NCA.models.LinearAverage import LinearAverage
from Contrastive_uncertainty.toy_NCA.models.NCA import NCACrossEntropy

class NCAToy(Toy):
    def __init__(self,
        datamodule,
        labels,
        margin :int = 0,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int = 20,
        emb_dim: int = 2,
        num_classes:int = 2,
        softmax_temperature:float = 0.07,
        memory_momentum: float = 0.05
        ):

        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodle = datamodule
        #import ipdb; ipdb.set_trace()
        
        ndata = len(self.datamodule.train_dataloader().dataset)
        self.lemniscate = LinearAverage(emb_dim, ndata, softmax_temperature, memory_momentum).cuda()
        self.criterion = NCACrossEntropy(labels, margin / softmax_temperature).cuda()

        
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = self.init_encoders()

    
    
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        return encoder
    
    def callback_vector(self,x):
        z = self.encoder(x)
        return z
    
    def loss_function(self, batch, auxillary_data=None):
        (data, *aug_data), labels, indices = batch

        features = self.encoder(data)
        outputs = self.lemniscate(features, indices)
        loss = self.criterion(outputs, indices)
        metrics = {'Loss': loss}

        #acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        #metrics = {'Loss': loss, 'Accuracy @ 1': acc1, 'Accuracy @5': acc5}
        return metrics     
    