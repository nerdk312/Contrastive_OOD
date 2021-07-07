import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from tqdm import tqdm
import faiss
import collections
import pytorch_lightning as pl

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.models.encoder_model import Backbone
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean



class HSupConCentroidToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 2,
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        num_negatives: int = 32,
        encoder_momentum: float = 0.999,
        pretrained_network:str = None,
        ):
        """
        hidden_dim: dimensionality of neural network (default: 128)
        emb_dim: dimensionality of the feature space (default: 2)
        num_negatives: number of negative samples/prototypes (defaul: 16384)

        encoder_momentum: momentum for updating key encoder (default: 0.999)
        softmax_temperature: softmax temperature
        """

        super().__init__()
        
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.num_channels = datamodule.num_channels

        
        self.encoder_q, self.encoder_k = self.init_encoders()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @property
    def name(self):
        ''' return name of model'''
        return 'HSupConCentroid'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(20, self.hparams.emb_dim)
        encoder_k = Backbone(20, self.hparams.emb_dim)
      
        return encoder_q, encoder_k

    def callback_vector(self, x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder_k(x)
        z = nn.functional.normalize(z, dim=1)
        return z
    
    def instance_vector(self, x):
        z = self.callback_vector(x)
        return z
   
    def fine_vector(self, x):
        z = self.callback_vector(x)
        return z

    def coarse_vector(self, x):
        z = self.callback_vector(x)
        return z

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
       
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # Nawid - add the keys to the queue
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer
        self.queue_ptr[0] = ptr

    # Calculate the centroids per epoch using the labels and the momentum encoder
    def calculate_centroids(self):
        pass
    
    # Loss between the centroid of a particular class and the data points in the same class 
    def alignment_loss(self):
    
    # Variance loss (maximise the variance) either globally or within the class by using the uniformity loss in the understanding contrastive learning as a point on a hypersphere or the vicreg loss
    def uniformity_loss(self):


    def loss_function(self, batch):
        metrics = {}
        # import ipdb; ipdb.set_trace()
        # *labels used to group together the labels
        (img_1, img_2), *labels, indices = batch
        #import ipdb; ipdb.set_trace()
        #collated_labels = [fine_labels,coarse_labels]
        
        q = self.encoder_q(img_1)
        k = self.encoder_k(img_2)

        q = self.encoder_q.sequential[0](q)
        instance_q = self.encoder_q.branch_fc[0](q)
        instance_q = nn.functional.normalize(instance_q, dim=1)

        k = self.encoder_q.sequential[0](k)
        instance_k = self.encoder_k.branch_fc[0](k)
        instance_k = nn.functional.normalize(instance_k, dim=1)
        
        output, target = self.instance_forward(instance_q,instance_k)
        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss
        acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        instance_metrics = {'Instance Loss': loss_instance, 'Instance Accuracy @1':acc_1,'Instance Accuracy @5':acc_5}
        metrics.update(instance_metrics)

        # Initialise loss value
        
        proto_loss_terms = [0, 0]
        assert len(proto_loss_terms) == len(labels), 'number of label types different than loss terms'
        for index, data_labels in enumerate(labels):
            
            q = self.encoder_q.sequential[index+1](q)
            proto_q = self.encoder_q.branch_fc[index+1](q)
            proto_q = nn.functional.normalize(proto_q, dim=1)

            k = self.encoder_q.sequential[index+1](k)
            proto_k = self.encoder_k.branch_fc[index+1](k)
            proto_k = nn.functional.normalize(proto_k, dim=1)

            features = torch.cat([proto_q.unsqueeze(1), proto_k.unsqueeze(1)], dim=1)
            proto_loss_terms[index] = self.supervised_contrastive_forward(features=features,labels=data_labels)
        
        # Normalise the proto loss by number of different labels present
        loss = (self.hparams.branch_weights[0]*loss_instance) + (self.hparams.branch_weights[1]*proto_loss_terms[0]) + (self.hparams.branch_weights[2]*proto_loss_terms[1])  # Nawid - increase the loss
        # import ipdb; ipdb.set_trace()
        additional_metrics = {'Loss':loss, 'Fine Loss':proto_loss_terms[0], 'Coarse Loss':proto_loss_terms[1]}
        metrics.update(additional_metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx,dataset_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
    
    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer
    
    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        