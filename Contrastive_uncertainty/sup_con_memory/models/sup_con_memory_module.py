import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.sup_con.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50

class SupConMemoryModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        contrast_mode:str = 'one',
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.num_channels = datamodule.num_channels

        # create the encoders
        # num_classes is the output fc dimension
        
        self.encoder_q, self.encoder_k = self.init_encoders()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # queue for the labels
        self.register_buffer("label_queue", torch.zeros(num_negatives))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        '''  
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)
        '''
    @property
    def name(self):
        ''' return name of model'''
        return 'SupConMemory'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder_q = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
            encoder_k = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder_q = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
            encoder_k = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
        
        return encoder_q, encoder_k

    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, key_labels):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # Nawid - add the keys to the queue
        
        # replace the label keys at ptr (dequeue and enqueue)
        self.label_queue[ptr:ptr + batch_size] = key_labels

        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer
        self.queue_ptr[0] = ptr
    

    
    def callback_vector(self, x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder(x)
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

    def forward(self,im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets, proto_logits, proto_targets
        """ 

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1) # Nawid - normalised key embeddings

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1) # Nawid - normalised query embeddings

        # Obtain the representations and the labels from the queue
        queue_keys, queue_labels = self.queue.clone().detach().T, self.label_queue.clone().detach()

        # Concatenate the momentum encoder outputs
        contrast_features = torch.cat([k,queue_keys],dim=0)
        # Concatenate the labels from the current iteration and the queue
        contrast_labels = torch.cat([labels,queue_labels],dim=0)
        # Calculate the supervised contrastive loss using the online encoder output, the momentum encoder output and the queue
        loss = self.supervised_contrastive_forward(q,contrast_features, labels, contrast_labels)

        self._dequeue_and_enqueue(k,labels) # Nawid - queue values

        return loss 
    
    def supervised_contrastive_forward(self,anchor_features, contrast_features, anchor_labels, contrast_labels, mask = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            anchor_features: hidden vector of shape [bsz, emb_dim]
            contrast_features: hidden vector of shape [bsz + num negatives, emb_dim]
            anchor_labels: ground truth of shape [bsz].
            contrast_labels: ground truth of shape [bsz].
            
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        anchor_bsz = anchor_features.shape[0]
        contrast_bsz = contrast_features.shape[0]
        
        # Based on answer given to my pytorch question https://discuss.pytorch.org/t/binary-2d-mask-implementation/128637
        mask = (anchor_labels[:,None] == contrast_labels[None,:])        
        anchor_dot_contrast = torch.div(  # Nawid - similarity between the anchor and the contrast feature , shape : [bsz, bsz +num_negatives]
            torch.matmul(anchor_features, contrast_features.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Nawid - logits mask is values of 0s and 1 in the same shape as the mask, it has values of 0 along the diagonal and 1 elsewhere, where the 0s are used to mask out self contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_bsz).view(-1, 1).to(self.device),
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
        loss = - (self.hparams.softmax_temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(1, anchor_bsz).mean()
        
        return loss




    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels

        loss = self(img_1, img_2, labels)
        metrics = {'Loss': loss}
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