import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from torch.utils import data
from tqdm import tqdm
import faiss
import collections
import pytorch_lightning as pl

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.models.encoder_model import Backbone
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.utils.pl_metrics import precision_at_k, mean

class HSupConBUCentroidOnlineToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 2,
        contrast_mode:str = 'one',
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        num_negatives: int = 32,
        encoder_momentum: float = 0.999,
        pretrained_network:str = None,
        branch_weights: list = [1.0/3, 1.0/3, 1.0/3],  # Going from instance fine to coarse
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

        # Classes for the case of the anchor and the positive
        self.anchor_classes = [datamodule.num_classes, datamodule.num_coarse_classes]
        self.positive_classes = [None, datamodule.num_classes] # None is used for the instance discrimination case
        self.coarse_mapping = self.datamodule.coarse_mapping # Coarse mapping is the mapping from the fine labels to the coarse labels
        # import ipdb; ipdb.set_trace()

        self.encoder_q, self.encoder_k = self.init_encoders()
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.centroid_q, self.centroid_k = self.init_centroids()
        '''
        #import ipdb; ipdb.set_trace()
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.pretrained_network)
        '''

    @property
    def name(self):
        ''' return name of model'''
        return 'HSupConBUCentroidOnline'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(20, self.hparams.emb_dim)
        encoder_k = Backbone(20, self.hparams.emb_dim)

        # Additional branches for the specific tasks
        fc_layer_dict = collections.OrderedDict([])
        Sequential_layer_dict = collections.OrderedDict([])

        fc_layer_dict['Instance'] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
        Sequential_layer_dict['Instance'] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
                
        for i in range(2):
            name = 'Proto_Fine' if i == 0 else 'Proto_Coarse'

            fc_layer_dict[name] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
            Sequential_layer_dict[name] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))

        encoder_q.sequential = nn.Sequential(Sequential_layer_dict)
        encoder_k.sequential = nn.Sequential(Sequential_layer_dict)

        encoder_q.branch_fc = nn.Sequential(fc_layer_dict)
        encoder_k.branch_fc = nn.Sequential(fc_layer_dict)
        
        return encoder_q, encoder_k
    
    def init_centroids(self):
        centroid_q_dict = {}
        centroid_k_dict = {}

        centroid_q_dict['fine'] = torch.normal(torch.zeros(self.datamodule.num_classes, self.hparams.emb_dim, requires_grad=True,device='cuda'), 1)
        centroid_k_dict['fine'] = torch.normal(torch.zeros(self.datamodule.num_classes, self.hparams.emb_dim,requires_grad=False,device='cuda'), 1)

        centroid_q_dict['coarse'] = torch.normal(torch.zeros(self.datamodule.num_coarse_classes, self.hparams.emb_dim,requires_grad=True,device='cuda'), 1)
        centroid_k_dict['coarse'] = torch.normal(torch.zeros(self.datamodule.num_coarse_classes, self.hparams.emb_dim,requires_grad=False,device='cuda'), 1)

        return centroid_q_dict, centroid_k_dict


    '''
    # Callback vector which uses both the representations for the task
    def callback_vector(self, x):  # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        coarse_z = self.coarse_callback_vector(x)
        fine_z = self.fine_callback_vector(x)
        z = (coarse_z, fine_z)
        return z
        
    # Callback vector for fine branch
    def fine_callback_vector(self, x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:2](z)
        z = self.encoder_k.branch_fc[1](z)
        z = nn.functional.normalize(z, dim=1)
        return z

    # Callback vector for coarse branch
    def coarse_callback_vector(self,x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:3](z)
        z = self.encoder_k.branch_fc[2](z)
        z = nn.functional.normalize(z, dim=1)
        return z
    '''

    def instance_vector(self, x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:1](z)
        z = self.encoder_k.branch_fc[0](z)
        z = nn.functional.normalize(z, dim=1)
        return z

    
    def fine_vector(self,x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:2](z)
        z = self.encoder_k.branch_fc[1](z)
        z = nn.functional.normalize(z, dim=1)
        return z

    def coarse_vector(self,x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:3](z)
        z = self.encoder_k.branch_fc[2](z)
        z = nn.functional.normalize(z, dim=1)
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

    def instance_forward(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #Nawid - positive logit between output of key and query
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # Nawid - negative logits (dot product between key and negative samples in a query bank)

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1) # Nawid - total logits - instance based loss to keep property of local smoothness

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long,device = self.device)
        
        #labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k) # Nawid - queue values
        
        return logits, labels
      
    def loss_function(self, batch):
        self._momentum_update_key_encoder()
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
        
        output, target = self.instance_forward(instance_q, instance_k)
        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target)  # Nawid - instance based info NCE loss
        acc_1, acc_5 = precision_at_k(output, target, top_k=(1,5))
        instance_metrics = {'Instance Loss': loss_instance, 'Instance Accuracy @1':acc_1,'Instance Accuracy @5':acc_5}
        metrics.update(instance_metrics)

        # Initialise loss value
        
        proto_loss_terms = [0, 0]
        # Mapping to use of indices to key words for the dict
        order_mapping = ['fine','coarse']
        #print('label length',len(labels))
        assert len(proto_loss_terms) == len(labels), 'number of label types different than loss terms'
        assert len(labels) == len(self.anchor_classes)
        for index, data_labels in enumerate(labels):
            
            q = self.encoder_q.sequential[index+1](q)
            proto_q = self.encoder_q.branch_fc[index+1](q) # Obtain representations which belong to the same class
            # Calculate local centroid
            proto_q = self.calculate_centroid(proto_q, data_labels, self.anchor_classes[index]) # calculates the centroids based on aggregating the samples in the same class
            
            # Update the centroid for the fine or the coarse by using the representation of the data
            self.centroid_q[order_mapping[index]] = self.hparams.encoder_momentum * self.centroid_q[order_mapping[index]] + (1 - self.hparams.encoder_momentum) * proto_q
            self.centroid_q[order_mapping[index]] = nn.functional.normalize(self.centroid_q[order_mapping[index]], dim=1) 
            proto_q = self.centroid_q[order_mapping[index]]
            #proto_q = nn.functional.normalize(proto_q, dim=1) # normalises the centroids
            

            k = self.encoder_q.sequential[index+1](k)
            proto_k = self.encoder_k.branch_fc[index+1](k)

            # Makes centroids for the particular class
            if self.positive_classes[index] is None:
                # Instance is the positives, no centroids made and labels is the data labels
                target_labels = data_labels.to(self.device)
                proto_k = nn.functional.normalize(proto_k, dim=1)
            else:
                #import ipdb; ipdb.set_trace()    
                proto_k = self.calculate_centroid(proto_k, labels[index-1], self.positive_classes[index])
                # Update the centroid k and normalise it, then set proto k as the new centroid
                self.centroid_k[order_mapping[index-1]] = self.hparams.encoder_momentum * self.centroid_k[order_mapping[index-1]] + (1 - self.hparams.encoder_momentum) * proto_k
                self.centroid_k[order_mapping[index-1]] = nn.functional.normalize(self.centroid_k[order_mapping[index-1]], dim=1) 
                proto_k = self.centroid_k[order_mapping[index-1]]
                # Label mapping from fine to coarse labels
                target_labels = self.coarse_mapping.to(self.device)
            
            logits = torch.einsum('fd,cd->fc', [proto_k, proto_q])  # Calculate the logits by computing dot product between (num class layer i-1 , dim) by (num class layer i, dim) to give (num class layer i-1, num class layer i)
            logits /= self.hparams.softmax_temperature
            
            # Computes cross entropy loss for the representation
            proto_loss_terms[index] = F.cross_entropy(logits, target_labels)  # Nawid - instance based info NCE loss
            
        
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

    
    def calculate_centroid(self, x, labels,no_classes): # Assume y is one hot encoder 
        y = F.one_hot(labels.long(), num_classes=no_classes).float()
        # compute sum of embeddings on class by class basis
        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        features_sum = torch.matmul(y.T,x) # (num_classes,batch) (batch,features) to get (num_class, features)

        #features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)

        embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings
    
    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        