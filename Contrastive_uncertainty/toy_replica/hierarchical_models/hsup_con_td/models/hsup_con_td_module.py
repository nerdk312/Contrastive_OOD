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
from Contrastive_uncertainty.general_hierarchy.utils.pl_metrics import precision_at_k, mean



class HSupConTDToy(pl.LightningModule):
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

        
        self.encoder_q, self.encoder_k = self.init_encoders()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        '''
        #import ipdb; ipdb.set_trace()
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.pretrained_network)
        '''

    @property
    def name(self):
        ''' return name of model'''
        return 'HSupConTD'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(20, self.hparams.emb_dim)
        encoder_k = Backbone(20, self.hparams.emb_dim)

        # Additional branches for the specific tasks
        fc_layer_dict = collections.OrderedDict([])
        Sequential_layer_dict = collections.OrderedDict([])

        for i in range(2):
            name = 'Proto_Coarse' if i == 0 else 'Proto_Fine'
            fc_layer_dict[name] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
            Sequential_layer_dict[name] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))


        fc_layer_dict['Instance'] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
        Sequential_layer_dict['Instance'] = nn.Sequential(nn.ReLU(), nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim))
                
        
        encoder_q.sequential = nn.Sequential(Sequential_layer_dict)
        encoder_k.sequential = nn.Sequential(Sequential_layer_dict)

        encoder_q.branch_fc = nn.Sequential(fc_layer_dict)
        encoder_k.branch_fc = nn.Sequential(fc_layer_dict)

        return encoder_q, encoder_k

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
    def instance_vector(self,x):
        z = self.encoder_k(x)
        z = self.encoder_k.sequential[0:3](z)
        z = self.encoder_k.branch_fc[2](z)
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
        z = self.encoder_k.sequential[0:1](z)
        z = self.encoder_k.branch_fc[0](z)
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
    
    def supervised_contrastive_forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #import ipdb; ipdb.set_trace()
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device = self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # Makes a mask with values of 0 and 1 depending on whether the labels between two different samples in the batch are the same
            mask = torch.eq(labels, labels.T).float().to(self.device)
            
        else:
            mask = mask.float().to(self.device)
        contrast_count = features.shape[1]
        # Nawid - concatenates the features from the different views, so this is the data points of all the different views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.hparams.contrast_mode == 'one':
            anchor_feature = features[:, 0] # Nawid - anchor is only the index itself and only the single view
            anchor_count = 1 # Nawid - only one anchor
        elif self.hparams.contrast_mode == 'all':
            anchor_feature = contrast_feature 
            anchor_count = contrast_count # Nawid - all the different views are the anchors
        else:
            raise ValueError('Unknown mode: {}'.format(self.hparams.contrast_mode))


        #anchor_feature = contrast_feature
        #anchor_count = contrast_count  # Nawid - all the different views are the anchors

        # compute logits (between each data point with every other data point )
        anchor_dot_con_bu_contrast = torch.div(  # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_con_bu_contrast, dim=1, keepdim=True)
        logits = anchor_dot_con_bu_contrast - logits_max.detach()
        # tile mask
        # Nawid- changes mask from [b,b] to [b* anchor count, b *contrast count]
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # Nawid - logits mask is values of 0s and 1 in the same shape as the mask, it has values of 0 along the diagonal and 1 elsewhere, where the 0s are used to mask out self contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        # Nawid - updates mask to remove self contrast examples
        mask = mask * logits_mask
        # compute log_prob
        # Nawid- exponentiate the logits and turn the logits for the self-contrast cases to zero
        exp_logits = torch.exp(logits) * logits_mask
        # Nawid - subtract the value for all the values along the dimension
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # Nawid - mask out all valeus which are zero, then calculate the sum of values along that dimension and then divide by sum
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        #loss = - 1 * mean_log_prob_pos
        #loss = - (model.hparams.softmax_temperature / model.hparams.base_temperature) * mean_log_prob_pos
        # Nawid - loss is size [b]
        loss = - (self.hparams.softmax_temperature / 0.07) * mean_log_prob_pos
        # Nawid - changes to shape (anchor_count, batch)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


    def loss_function(self, batch):
        # Encoder update
        self._momentum_update_key_encoder()
        metrics = {}
        (img_1, img_2), *labels, indices = batch
        
        q = self.encoder_q(img_1)
        k = self.encoder_k(img_2)
        # Proto loss
        proto_loss_terms = [0, 0]
        assert len(proto_loss_terms) == len(labels), 'number of label types different than loss terms'
        for index, data_labels in enumerate(labels):
            
            q = self.encoder_q.sequential[index](q)
            proto_q = self.encoder_q.branch_fc[index](q)
            proto_q = nn.functional.normalize(proto_q, dim=1)

            k = self.encoder_q.sequential[index](k)
            proto_k = self.encoder_k.branch_fc[index](k)
            proto_k = nn.functional.normalize(proto_k, dim=1)

            features = torch.cat([proto_q.unsqueeze(1), proto_k.unsqueeze(1)], dim=1)
            proto_loss_terms[index] = self.supervised_contrastive_forward(features=features,labels=data_labels)
        

        # Instance loss
        q = self.encoder_q.sequential[-1](q)
        instance_q = self.encoder_q.branch_fc[-1](q)
        instance_q = nn.functional.normalize(instance_q, dim=1)

        k = self.encoder_q.sequential[-1](k)
        instance_k = self.encoder_k.branch_fc[-1](k)
        instance_k = nn.functional.normalize(instance_k, dim=1)
        
        output, target = self.instance_forward(instance_q,instance_k)
        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss
        acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        instance_metrics = {'Instance Loss': loss_instance, 'Instance Accuracy @1':acc_1,'Instance Accuracy @5':acc_5}
        metrics.update(instance_metrics)

        # Total loss
        loss = (self.hparams.branch_weights[0]*loss_instance) + (self.hparams.branch_weights[1]*proto_loss_terms[0]) + (self.hparams.branch_weights[2]*proto_loss_terms[1])  # Nawid - increase the loss

        # Metric logging
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

        