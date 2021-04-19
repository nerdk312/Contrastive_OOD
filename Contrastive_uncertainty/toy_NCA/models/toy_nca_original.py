import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import math
import numpy as np

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k

from Contrastive_uncertainty.toy_NCA.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_NCA.models.toy_module import Toy



class NCAOriginalToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int = 20,
        emb_dim: int = 2,
        num_classes:int = 2,
        ):

        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodle = datamodule
        #import ipdb; ipdb.set_trace()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = self.init_encoders()

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        return encoder
    
    # Obtain feature vector
    def callback_vector(self, x):
        z = self.encoder(x)
        return z
    
    def loss_function(self, batch, auxillary_data=None):
        (data, *aug_data), labels, indices = batch
        
        #_, labels = torch.max(labels,dim=1)

        distances = self.encoder.original_dists(data)  # Obtain the distance of the class
        distances.diagonal().copy_(np.inf*torch.ones(len(distances)))

        # compute pairwise probability matrix p_ij defined by a softmax over negative squared distances in the transformed space.
        # since we are dealing with negative values with the largest value being 0, we need  not worry about numerical instabilities
        # in the softmax function
        p_ij = self.encoder._softmax(-distances)

        # compute pairwise boolean class matrix
        y_mask = labels[:, None] == labels[None, :]

        # for each p_i, zero out any p_ij that  is not of the same class label as i
        p_ij_mask = p_ij * y_mask.float()
        
        # sum over js to compute p_i
        p_i = p_ij_mask.sum(dim=1)

        accuracy = len(p_i[p_i>0.5])/len(p_i) # obtain the number of elements which are above 0.5 for the predictions and the total number of elements to get the accuracy

        accuracy = torch.tensor([accuracy])

        # compute expected number of points correctly classified by summing over all p_i's.
        # to maximize the above expectation we can negate it and feed it to a minimizer
        # for numerical stability, we only log_sum over non-zero values
        classification_loss = -torch.log(torch.masked_select(p_i, p_i != 0)).sum()
        
        # to prevent the embeddings of different classes from collapsing to the same point, we add a hinge loss penalty
        distances.diagonal().copy_(torch.zeros(len(distances)))
        margin_diff = (1 - distances) * (~y_mask).float()
        hinge_loss = torch.clamp(margin_diff, min=0).pow(2).sum(1).mean()


        # sum both loss terms and return
        loss = classification_loss + hinge_loss

        metrics = {'Loss': loss,'Accuracy':accuracy, 'Classification Loss':classification_loss, 'Hinge Loss':hinge_loss}

        return metrics 