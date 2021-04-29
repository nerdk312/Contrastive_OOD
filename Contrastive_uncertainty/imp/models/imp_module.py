import numpy as np
from random import sample
import wandb

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from tqdm import tqdm
import faiss
import math


from Contrastive_uncertainty.imp.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.imp.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.imp.callbacks.general_callbacks import quickloading
#from Contrastive_uncertainty.imp.utils.imp_utils import 

class base_module(pl.LightningModule):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.datamodule = datamodule # Used for the purpose of obtaining data loader for the case of epoch starting
                
        
    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        #import ipdb; ipdb.set_trace()
        for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)

    def loss_function(self, batch, auxillary_data=None):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer


class IMPModule(base_module):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        emb_dim: int = 128,
        alpha: float = 1.0,
        init_sigma: float = 1.0,
        use_mlp: bool = True,
        num_channels:int = 3, # number of channels for the specific dataset
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None):


        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        

        # Sigma parameter related to obtaining the clusters
        log_sigma = torch.log(torch.FloatTensor([sigma]))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma, requires_grad=True)
        else:
            self.log_sigma = log_sigma

        self.encoder = self.init_encoders()
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()
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
            encoder_q = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = 10)
            
        return encoder

    def callback_vector(self,x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder(x)
        z = nn.functional.normalize(z, dim=1)
        return z
                
    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Nawid - calculate prototypes (may need to fix)
    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [N, D] encoded inputs
            probs: [N, nClusters] soft assignment
        Returns:
            cluster protos: [ nClusters, D]
        """

        h = torch.unsqueeze(h, 1)       # [N, 1, D]
        probs = torch.unsqueeze(probs, 2)       # [N, nClusters, 1]
        prob_sum = torch.sum(probs, 0)  # [nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        protos = h*probs    # [N, nClusters, D]
        protos = torch.sum(protos, 0)/prob_sum # [nClusters,D]
        return protos

    def _add_cluster(self, nClusters, protos, radii):
        """
        Args:
            nClusters: number of clusters
            protos: [nClusters, D] cluster protos
            radii: [nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1

        d_radii = torch.ones(1).cuda()

        # Nawid - sigma l is the labelled cluster variance whilst sigma u is the unlabelled cluster variance
        d_radii = d_radii * torch.exp(self.log_sigma_u)
        # Makes a new prototype from the example
        new_proto = ex.unsqueeze(0).cuda()
        # Nawid - obtain new prototypes and radii
        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii
    
    # Nawid - estimate lambda (distance threshold for creating new cluster)
    # Need to fix the code for lambda
    def estimate_lambda(self, tensor_proto):
        # estimate lambda by mean of shared sigmas
        rho = tensor_proto[0].var(dim=0)
        rho = rho.mean()
        # Nawid- estimate sigma based on supervised or unsupervised

        sigma = torch.exp(self.log_sigma_u).data[0]
        # Nawid - obtain lambda hyperparameter and then calculate lambda (eq 5 in paper)
        lamda = -2*sigma*np.log(self.hparams.alpha) + self.hparams.emb_dim*sigma*np.log(1+rho/sigma)

        return lamda
    
    # Nawid - delete empty clusters
    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = torch.sum(prob[0],dim=0).data # check the probabilities to see what prototypes are being used
        good_protos = column_sums > eps # obtain good prototypes where probability is above a threshold near zero
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs] # Select the good prototypes and values



    # Nawid - calculate the loss using the closest prototype in the class , where N is the different number of prototpyes in a class I assume and nclusters is the number of classes
    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.
        inputs:
            logits [N, nClusters] of nll probs for each cluster
            targets [N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        target_logits = torch.ones_like(logits.data) * float('-Inf')
        
        # Nawid - I believe this calculates the logits for a particular class
        target_logits[targets] = logits.data[targets] # Nawid - select the specific logits by selecting the logits corresponding to the target
        # Nawid -choose the best targets using a max
        #_, best_targets = torch.max(target_logits, dim=1)
        # mask out everything...
        weights = torch.zeros_like(logits.data)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l
            class_logits = torch.ones_like(logits.data) * float('-Inf')
            class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data.view(logits.size(0), -1)
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[range(0, targets.size(0)), best_in_class] = 1.
        # Nawid - perform weighted cross entropy between targets and the weights (which are the best logits in a particular class)
        loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        return loss.mean()

    # Nawid - forward pass for unsupervised data, creates clusters in unsupervised way
    def forward(self, samples, unlabel_lambda=20., num_cluster_steps=5):
        h = self.encoder(samples)
        h = nn.functional.normalize(h, dim=1)
        
        for ii in range(num_cluster_steps):
            for i, ex in enumerate(h[0]):
                distances = self._compute_distances(tensor_proto, ex.data)
                if (torch.min(distances) > unlabel_lambda):
                    nClusters, tensor_proto, radii = self._add_cluster(nClusters, tensor_proto, radii, 'labeled', ex.data)
            
            # Nawid - assign cluster radii
            prob_unlabel = assign_cluster_radii(Variable(tensor_proto).cuda(), h_unlabel, radii)
            prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
            prob_all = prob_unlabel_nograd
            protos = self._compute_protos(h_all, prob_all)
        return {'logits':prob_unlabel_nograd.data[0]}


    # Nawid - returns variance of a particular class
    def _embedding_variance(self, x):
        """Compute variance in embedding space
        Args:
            x: examples from one class
        Returns:
            in-class variance
        """
        h = self.encoder(x)   # [N, D]
        h = nn.functional.normalize(h, dim=1)
        #D = h.size()[1]
        #h = h.view(-1, D)   # [N, D]
        variance = torch.var(h, 0)
        return torch.sum(variance)
    
    # Nawid - I believe x_list is a list corresponding to data points of each class
    def _within_class_variance(self, x_list):
        protos = []
        for x in x_list:
            h = self.encoder(x)
            h = nn.functional.normalize(h, dim=1)
            #D = h.size()[2]
            #h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0) # Nawid - calculate mean of a class
            protos.append(proto) # calculate 
        protos = torch.cat(protos, 0) # concatenate all prototypes
        variance = torch.var(protos, 0) # Calculate the variance for each value
        return torch.sum(variance)
    
    def _within_class_distance(self, x_list):
        protos = []
        for x in x_list:
            h = self.encoder(x)
            h = nn.functional.normalize(h, dim=1)
            #D = h.size()[2]
            #h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0, keepdim=True) # [1,D]
            protos.append(proto)
        protos = torch.cat(protos, 0).data.cpu().numpy()   # [C, D]
        num_classes = protos.shape[0]
        distances = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i > j:
                    dist = np.sum((protos[i, :] - protos[j, :])**2)
                    distances.append(dist)
        return np.mean(distances)
    
    # Nawid - returns number of elements in each cluster based on a soft cluster assignment
    def _get_count(self, probs, soft=True):
        """
        Args:
            probs: [B, N, nClusters] soft assignments
        Returns:
            counts: [B, nClusters] number of elements in each cluster
        """
        if not soft:
            _, max_indices = torch.max(probs, 1)    # [N]
            nClusters = probs.size()[1] # [nclusters]
            # NEED TO FIX FROM THIS POINT
            max_indices = one_hot(max_indices, nClusters)
            counts = torch.sum(max_indices, 1).cuda()
        else:
            counts = torch.sum(probs, 1)
        return counts

        