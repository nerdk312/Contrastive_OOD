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


class VAEModule(pl.LightningModule):
    def __init__(self,
        instance_encoder:str ='resnet18',
        first_conv:bool= False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        emb_dim: int = 256,
        kl_coeff: float = 0.1,
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
        self.encoder, self.decoder = self.init_encoders()
        self.fc_mu  = nn.Linear(self.enc_out_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.emb_dim)
    
    @property
    def name(self):
        ''' return name of model'''
        return 'VAE'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder  = resnet18_encoder(self.num_channels,self.enc_out_dim, self.first_conv, self.maxpool1)
            decoder = resnet18_decoder(self.num_channels, self.emb_dim, self.input_height, self.first_conv, self.maxpool1)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder = resnet50_encoder(self.num_channels, self.enc_out_dim, self.first_conv, self.maxpool1)
            decoder = resnet50_decoder(self.num_channels, self.emb_dim, self.input_height, self.first_conv, self.maxpool1)

        return encoder, decoder
    
    def vae_forward(self, x):
        x = self.encoder(x)
        # Added the normalisation myself
        #x = nn.functional.normalize(x, dim=1)
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
        x = self.encoder(x)
        # Added the normalisation myself
        #x = nn.functional.normalize(x, dim=1)
        #import ipdb; ipdb.set_trace()
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q


    def callback_vector(self, x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        
        z = self.encoder(x)
        #z = nn.functional.normalize(z, dim=1)
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
        (img_1, _), labels, indices = batch
        

        z, x_hat, p, q = self._run_step(img_1)
        #import ipdb; ipdb.set_trace()
        recon_loss = F.mse_loss(x_hat, img_1, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.kl_coeff

        loss = kl + recon_loss

        logs = {
            "Reconstruction Loss": recon_loss,
            "KL": kl,
            "Loss": loss,
        }
        metrics.update(logs)
        #print('Loss',loss)
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