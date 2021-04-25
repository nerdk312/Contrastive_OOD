import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.npid_pcl.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.npid_pcl.models.simple_memory import SimpleMemory
from Contrastive_uncertainty.npid_pcl.utils.pl_metrics import precision_at_k

# Based on code from https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/npid_pcl.py
class npid_pclModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        num_negatives: int = 8192,
        softmax_temperature: float = 0.07,
        memory_momentum = 0.5,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodule = datamodule

        # Instantiate memory bank
        self.memory_bank = SimpleMemory(length=self.datamodule.total_samples,feat_dim=emb_dim, momentum=memory_momentum)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder= self.init_encoders()
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

    
    def forward(self, img, idx):
        """ Forward computation

        Args:
        img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
        
        Returns:
        logits (Tensor): logits of the data

        """
        features = self.encoder(img) 
        features = nn.functional.normalize(features) # BxD
        bs, feat_dim = features.shape[:2]
        neg_idx = self.memory_bank.multinomial.draw(bs * self.hparams.num_negatives)

        # Obtain positive features 
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC

        # Obtain negative features
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.hparams.num_negatives,
                                                    feat_dim)  # BxKxC
        
        # Obtain positive and negative logits of the data
        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, features]).unsqueeze(-1)
        neg_logits = torch.bmm(neg_feat, features.unsqueeze(2)).squeeze(2)


        logits = torch.cat([pos_logits, neg_logits], dim=1)
        # apply temperature
        logits /= self.hparams.softmax_temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long) # Nawid - class zero is always the correct class, which corresponds to the postiive examples in the logit tensor (due to the positives being concatenated first)
        labels = labels.type_as(logits)

        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, features.detach())

        return logits, labels


    def loss_function(self, batch):
        (img_1, img_2), labels,indices = batch
        output, target = self(img=img_1,idx=indices)      
        loss = F.cross_entropy(output, target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        metrics = {'Loss': loss, 'Instance Accuracy @ 1': acc1, 'Instance Accuracy @ 5': acc5}

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
