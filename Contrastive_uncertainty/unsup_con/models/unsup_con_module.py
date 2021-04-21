import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.unsup_con.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50

from Contrastive_uncertainty.general_clustering.models.pcl_module import datasize, \
    compute_features , run_kmeans, cluster_data, \
    on_train_epoch_start, on_fit_start, aux_data
      

class UnSupConModule(pl.LightningModule):
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


    def forward(self, features, labels=None, mask=None):
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
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        contrast_count = features.shape[1]
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

        # compute logits
        anchor_dot_contrast = torch.div(  # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        #loss = - 1 * mean_log_prob_pos
        #loss = - (model.hparams.softmax_temperature / model.hparams.base_temperature) * mean_log_prob_pos
        loss = - (self.hparams.softmax_temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


    def loss_function(self, batch,cluster_result):
        metrics = {}
        loss = torch.tensor([0],device = self.device)
        (img_1, img_2), _, indices = batch
        # Obtain clustering labels
        for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
            pseudo_labels= im2cluster[indices] # Nawid - get the true cluster assignment for each of the different samples

            imgs = torch.cat([img_1, img_2], dim=0)
            bsz = pseudo_labels.shape[0]
            features = self.encoder(imgs)
            features = nn.functional.normalize(features, dim=1)
            ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
            loss_proto = self(features, pseudo_labels) #  forward pass of the model

            # update metrics with the metrics for each cluster
            proto_metrics = {f'Proto Loss Cluster {self.hparams[n]}':loss_proto}
            metrics.update(proto_metrics)
            
            loss += loss_proto
        
        loss = loss/len(self.hparams.num_cluster)

        loss_metrics = {'Loss': loss}

        metrics.update(loss_metrics)
        return metrics

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
