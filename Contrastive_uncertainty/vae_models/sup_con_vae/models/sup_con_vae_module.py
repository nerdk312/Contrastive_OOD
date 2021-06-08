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


class SupConVAEModule(pl.LightningModule):
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
        contrast_mode:str = 'one',
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
        self.encoder, self.decoder = self.init_encoders()
        self.fc_mu  = nn.Linear(self.enc_out_dim , self.emb_dim)
        self.fc_var = nn.Linear(self.enc_out_dim , self.emb_dim)
    
    @property
    def name(self):
        ''' return name of model'''
        return 'SupConVAE'

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
        
        encoder.projection_head = nn.Linear(self.enc_out_dim, self.emb_dim) # Projection head for supervisec contrastive learning

        return encoder, decoder
    
    def vae_forward(self, x):
        x = self.encoder(x)
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
        x = self.encoder(x)
        # Added the normalisation myself
        x = nn.functional.normalize(x, dim=1)
        #import ipdb; ipdb.set_trace()
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q
    
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
        anchor_dot_contrast = torch.div(  # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
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
    
    

    # Pass in a latent representations and obtain reconstructed output
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def contrastive_representations(self, x):
        x = self.encoder(x)
        x = nn.functional.normalize(x, dim=1)
        z = self.encoder.projection_head(x)
        z = nn.functional.normalize(z, dim=1)
        return z 
        
    def loss_function(self, batch):
        

        metrics = {}
        # Sup con loss
        (img_1, img_2), labels,indices = batch
        imgs = torch.cat([img_1, img_2], dim=0)
        bsz = labels.shape[0]
        features = self.contrastive_representations(imgs)
        ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
        sup_con_loss = self.supervised_contrastive_forward(features, labels) #  forward pass of the model
        contrastive_metrics = {'SupCon Loss': sup_con_loss}
        metrics.update(contrastive_metrics)

        # VAE loss terms
        z, x_hat, p, q = self._run_step(img_1)
        #import ipdb; ipdb.set_trace()
        recon_loss = F.mse_loss(x_hat, img_1, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.kl_coeff

        loss = kl + recon_loss + sup_con_loss # Add the sup con loss also

        logs = {
            "Reconstruction Loss": recon_loss,
            "KL": kl,
            "Loss": loss,
        }
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