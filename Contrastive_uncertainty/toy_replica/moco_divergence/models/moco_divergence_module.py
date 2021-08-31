from numpy.core.fromnumeric import sort
from torch.utils import data
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from Contrastive_uncertainty.toy_replica.moco.models.encoder_model import Backbone
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean



class MocoDivergenceToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        weighting:float = 1.0,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
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

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def name(self):
        ''' return name of model'''
        return 'MocoDivergence'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(20, self.hparams.emb_dim)
        encoder_k = Backbone(20, self.hparams.emb_dim)
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
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # Nawid - add the keys to the queue
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer
        self.queue_ptr[0] = ptr

       
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


    def forward(self, im_q, im_k):
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
        return logits, labels, q

    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        
        output, target, q = self(img_1, img_2)
        loss_kl = self.kl_loss(q,labels)
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss

        # weighting used to weight the losses
        #print('weighting', self.hparams.weighting)
        loss =  (1- self.hparams.weighting)*loss_instance - (self.hparams.weighting*loss_kl) # The aim should be to maximise the KL divergence therefore I should be put a negative value in front 
        acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        metrics = {'Loss': loss, 'Loss KL':loss_kl, 'Loss Instance':loss_instance, 'Accuracy @1':acc_1,'Accuracy @5':acc_5}
        return metrics

    def kl_loss(self, query, labels):
        # Obtain representations for the different classes
        class_query = [query[labels==i] for i in torch.unique(labels,sorted=True)]
        num_classes = len(class_query)
        class_means = [torch.mean(class_query[class_num],axis=0) for class_num in range(num_classes)]
        class_std = [torch.std(class_query[class_num], axis=0) for class_num in range(num_classes)]
        class_gaussians = [torch.distributions.normal.Normal(class_means[class_num],class_std[class_num]) for class_num in range(num_classes)]
        class_KLs = [torch.mean(torch.distributions.kl.kl_divergence(self.total_distribution,class_gaussians[class_num])) for class_num in range(num_classes)]
        loss_KL = torch.mean(torch.stack(class_KLs))        

        return loss_KL


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
    
    # Calculate total distribution using the validation dataloader every epoch
    def on_train_epoch_start(self):
        dataloader = self.datamodule.val_dataloader() 
        if isinstance(dataloader,list) or isinstance(dataloader,tuple):
            _, dataloader = dataloader # obtain the version of the dataloader which does not use augmentations
        # calculate the total distribution
        self.total_distribution = self.compute_total_distribution(dataloader)
    
    def on_fit_start(self):
        dataloader = self.datamodule.val_dataloader() 
        if isinstance(dataloader,list) or isinstance(dataloader,tuple):
            _, dataloader = dataloader # obtain the version of the dataloader which does not use augmentations
        # calculate the centroid
        self.total_distribution = self.compute_total_distribution(dataloader)

    def compute_total_distribution(self,dataloader):

        #features = torch.zeros(self.datasize(dataloader), self.hparams.emb_dim, device = self.device)
        features = []
        for i, (images, *labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder_k(images)  # Nawid - obtain features for the task
            feat = nn.functional.normalize(feat, dim=1) # Obtain 12 normalised features for clustering
            
            features.append(feat)

        features = torch.cat(features,axis=0)
        features_mean = torch.mean(features,axis=0)
        features_std = torch.std(features,axis=0)
        total_distribution = torch.distributions.normal.Normal(features_mean,features_std) 
        return total_distribution



    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder_q.load_state_dict(checkpoint['target_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])
