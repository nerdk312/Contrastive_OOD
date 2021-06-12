from Contrastive_uncertainty.vae_models.vae.models.components import resnet18_encoder, resnet50_decoder
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.vae_models.vae.models.components import resnet18_encoder,resnet18_decoder, resnet50_encoder, resnet50_decoder
from Contrastive_uncertainty.general.utils.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k,mean


class MocoVAEModule(pl.LightningModule):
    def __init__(self,
        instance_encoder:str ='resnet18',
        first_conv:bool= False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        emb_dim: int = 128,
        kl_coeff: float = 0.1,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        pretrained_network:str = None,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()

        self.datamodule = datamodule
        self.num_channels = datamodule.num_channels
        self.num_classes = datamodule.num_classes

        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.emb_dim = emb_dim
        self.input_height = datamodule.input_height
        self.first_conv = first_conv
        self.maxpool1 = maxpool1


        
 
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q,self.encoder_k, self.decoder = self.init_encoders()
        self.fc_mu  = nn.Linear(self.enc_out_dim , self.emb_dim)
        self.fc_var = nn.Linear(self.enc_out_dim , self.emb_dim)



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
        return 'MocoVAE'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder_q  = resnet18_encoder(self.num_channels,self.enc_out_dim, self.first_conv, self.maxpool1)
            encoder_k  = resnet18_encoder(self.num_channels,self.enc_out_dim, self.first_conv, self.maxpool1)
            decoder = resnet18_decoder(self.num_channels, self.emb_dim, self.input_height, self.first_conv, self.maxpool1)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder_q = resnet50_encoder(self.num_channels, self.enc_out_dim, self.first_conv, self.maxpool1)
            encoder_k = resnet50_encoder(self.num_channels, self.enc_out_dim, self.first_conv, self.maxpool1)
            decoder = resnet50_decoder(self.num_channels, self.emb_dim, self.input_height, self.first_conv, self.maxpool1)

        encoder_q.projection_head = nn.Linear(self.enc_out_dim, self.emb_dim) # Projection head for supervisec contrastive learning
        encoder_k.projection_head = nn.Linear(self.enc_out_dim, self.emb_dim) # Projection head for supervisec contrastive learning

        return encoder_q, encoder_k, decoder
    

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


    def vae_forward(self, x):
        x = self.encoder_q(x)
        # Added the normalisation myself
        x = nn.functional.normalize(x, dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def _run_step(self, x):
        x = self.encoder_q(x)
        # Added the normalisation myself
        x = nn.functional.normalize(x, dim=1)
        #import ipdb; ipdb.set_trace()
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    
    def moco_forward(self, im_q, im_k):
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
            k = self.encoder_k.projection_head(k)
            k = nn.functional.normalize(k, dim=1)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1) # Nawid - normalised query embeddings
        q = self.encoder_q.projection_head(q)
        q = nn.functional.normalize(q, dim=1)
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
    
    

    # Pass in a latent representations and obtain reconstructed output
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def loss_function(self, batch):
        metrics = {}
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        
        output, target = self.moco_forward(img_1, img_2)
        moco_loss = F.cross_entropy(output, target) # Nawid - instance based info NCE loss
        acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        contrastive_metrics = {'Instance Discrimination Loss': moco_loss, 'Accuracy @1':acc_1,'Accuracy @5':acc_5}
        metrics.update(contrastive_metrics)

        z, x_hat, p, q = self._run_step(img_1)
        #import ipdb; ipdb.set_trace()
        recon_loss = F.mse_loss(x_hat, img_1, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.kl_coeff

        loss = kl + recon_loss + moco_loss

        logs = {
            "Reconstruction Loss": recon_loss,
            "KL": kl,
            "Loss": loss,
        }
        #print('Loss',loss)
        metrics.update(logs)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx, dataset_idx):
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
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])