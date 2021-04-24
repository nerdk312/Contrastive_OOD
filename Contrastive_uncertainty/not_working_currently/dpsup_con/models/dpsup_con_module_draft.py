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

from Contrastive_uncertainty.dpsup_con.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50


      

class DPSupConModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        contrast_mode:str = 'one',
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        use_mlp: bool = False,
        num_cluster: list = [10], # Clusters for training
        num_channels:int = 3, # number of channels for the specific dataset
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
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
        
        self.auxillary_data = None # Basic instantiation before model starts training


    # Sigma term for learning the clustering variance
        self.sigma = nn.Parameter(torch.zeros(num_classes))
        log_sigma_l = torch.log(torch.FloatTensor([config.init_sigma_l]))
        if config.learn_sigma_l:
            self.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True)

    
    def estimate_lambda(self, tensor_proto):
        # estimate lambda by mean of shared sigmas
        rho = tensor_proto[0].var(dim=0) # Density measured by variance between the different prototype vectors
        rho = rho.mean() # Calculate a single value from the different dimensions
        # Nawid- estimate sigma based on supervised or unsupervised
        
        sigma = torch.exp(self.log_sigma_l).data[0]
        # Nawid - obtain lambda hyperparameter and then calculate lambda (eq 5 in paper)
        lamda = -2*sigma*np.log(self.config.ALPHA) + self.config.dim*sigma*np.log(1+rho/sigma)

        return lamda

    # Nawid - calculate distance
    def _compute_distances(self, protos, example):
        dist = torch.sum((example - protos)**2, dim=2)
        return dist
        
    # Nawid - changed code to take into account single modality
    def _add_cluster(self, nClusters, protos, radii, ex = None):
        """
        Args:
            nClusters: number of clusters
            protos: [nClusters, D] cluster protos
            radii: [nClusters] cluster radius,
            
        Returns:
            updated arguments
        """
        
        nClusters += 1
        # Nawid-  batch size
        bsize = protos.size()[0]
        # Nawid - dimensionality of the embedding
        dimension = protos.size()[2]


        

        # Nawid - sigma l is the labelled cluster variance whilst sigma u is the unlabelled cluster variance
        d_radii = Variable(torch.ones(1), requires_grad=False).cuda()
        d_radii = d_radii * torch.exp(self.log_sigma_l)
        
        # Nawid - create new prototype using the added example
        new_proto = ex.unsqueeze(0).cuda()

        # Nawid - obtain new prototypes and radii
        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        
        return nClusters, protos, radii
    
    # Nawid - compute logits of being in a cluster based on euclidean distance
def compute_logits(cluster_centers, data):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
        cluster_centers: [K, D] Cluster center representation.
        data: [N, D] Data representation.
    Returns:
        log_prob: [N, K] logits.
    """
    cluster_centers = torch.unsqueeze(cluster_centers, 1)  # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    # [B, N, K]
    neg_dist = -torch.sum((data - cluster_centers)**2, 3)
    return neg_dist


    
    # Nawid - returns number of elements in each cluster based on a soft cluster assignment

    # Nawid - delete empty clusters
    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = torch.sum(prob[0],dim=0).data
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]

    



    
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



    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch,self.auxillary_data)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch,self.auxillary_data)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch,self.auxillary_data)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        
        #return {'logits':logits,'target':labels} # returns y_pred as y_pred are essentially the logits in this case, and I want to log how the logits change in time
     
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

    


    # Nawid - calculate prototypes
    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [N, D] encoded inputs
            probs: [N, nClusters] soft assignment
        Returns:
            cluster protos: [nClusters, D]
        """

        h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        protos = h*probs    # [B, N, nClusters, D]
        protos = torch.sum(protos, 1)/prob_sum
        return protos

    def forward(self,batch):
        # initialise single prototype centred at zero with a certain variance



        # calculate representation of data
        z = .....

        # compute prototypes from the soft assignment and the datapoint
        protos = self._compute_protos(z,probs)

        for ii in range(self.config.num_cluster_steps):
            # iterate through each representation
            for i, ex in enumerate(z):
                # Nawid - get indices corresponding to the support labels
                idxs = torch.nonzero(batch.y_train.data[0, i] == support_labels)[0]
                # Nawid - calculate distances
                distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)

                if (torch.min(distances) > lamda):
                    nClusters, tensor_proto, radii  = self._add_cluster(nClusters, tensor_proto, radii, ex=ex.data)



