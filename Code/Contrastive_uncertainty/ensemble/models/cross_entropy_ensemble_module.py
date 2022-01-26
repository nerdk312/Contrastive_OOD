import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.cross_entropy.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.general.utils.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k,mean


class CrossEntropyEnsembleModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        label_smoothing:bool = False,
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        num_models:int = 3
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        

        self.emb_dim = emb_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.instance_encoder = instance_encoder
        self.pretrained_network = pretrained_network
        self.num_models = num_models


        self.datamodule = datamodule
        self.num_channels = datamodule.num_channels
        self.num_classes = datamodule.num_classes
        print('num clases', self.num_classes)
        # create the encoders
        # num_classes is the output fc dimension
        self.encoders = nn.ModuleList(self.init_encoders())
        '''
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)
            print('loaded model')
        '''
    @property
    def name(self):
        ''' return name of model'''
        return 'CrossEntropyEnsemble'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.instance_encoder == 'resnet18':
            print('using resnet18')
            encoders  = [custom_resnet18(latent_size = self.emb_dim,num_channels = self.num_channels,num_classes=self.num_classes) for i in range(self.num_models)]
            
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoders = [custom_resnet50(latent_size = self.emb_dim,num_channels = self.num_channels,num_classes=self.num_classes) for i in range(self.num_models)]

        for i in range(self.num_models):
            encoders[i].class_fc2 = nn.Linear(self.emb_dim, self.num_classes)

        return encoders

    
    
    # parameter i corresponds to the model of interest
    def class_forward(self, x, i):
        z = self.encoders[i](x)
        z = nn.functional.normalize(z, dim=1)
        logits = self.encoders[i].class_fc2(z)
        return logits

    def loss_function(self, batch):
        metrics = {}
        
        
        (img_1, img_2), *labels, indices= batch
        # Takes into account if it has coarse labels
        # Using * makes it into a list (so the length of the list is related to how many different labels types there are)
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels

        '''
        if len(labels) > 1:
            labels = labels[0]
        '''
        # Loss from each of the independent models

        total_loss = 0
        # iterate through the different models
        for i in range(self.num_models):
            logits = self.class_forward(img_1, i)
            if self.label_smoothing:
                loss = LabelSmoothingCrossEntropy(ε=0.1, reduction='none')(logits.float(),labels.long()) 
                loss = torch.mean(loss)
            else:
                
                loss = F.cross_entropy(logits.float(), labels.long())
            
            class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
            individual_metrics = {f'Model {i} Loss': loss, f'Model {i} Class Accuracy @ 1': class_acc1, f'Model {i} Class Accuracy @ 5': class_acc5}

            total_loss += loss
            metrics.update(individual_metrics)
        
        Total_metrics = {'Total Loss': total_loss}
        metrics.update(Total_metrics)
        #metrics = {'Class Loss': total_loss}
        

        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Total Loss']
        return loss
        

    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
        

    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        
        
        #return {'logits':logits,'target':labels} # returns y_pred as y_pred are essentially the logits in this case, and I want to log how the logits change in time
     
    def configure_optimizers(self):
        if self.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.learning_rate,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.learning_rate,
                                        weight_decay = self.weight_decay)
        return optimizer
    
    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])