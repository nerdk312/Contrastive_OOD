import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.toy_replica.cross_entropy.models.encoder_model import Backbone
from Contrastive_uncertainty.general.utils.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy 
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean





class CrossEntropyToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        label_smoothing:bool = False,
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
        self.encoder = self.init_encoders()
        '''
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)
            print('loaded model')
        '''
    @property
    def name(self):
        ''' return name of model'''
        return 'CrossEntropy'

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(20, self.hparams.emb_dim)
        encoder.class_fc2 = nn.Linear(self.hparams.emb_dim, self.num_classes)
        return encoder

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
    
    def class_forward(self, x):
        z = self.encoder(x)
        z = nn.functional.normalize(z, dim=1)
        logits = self.encoder.class_fc2(z)
        return logits

    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        # Takes into account if it has coarse labels
        # Using * makes it into a list (so the length of the list is related to how many different labels types there are)
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        '''
        if len(labels) > 1:
            labels = labels[0]
        '''
        
        logits = self.class_forward(img_1)
        if self.hparams.label_smoothing:
            loss = LabelSmoothingCrossEntropy(ε=0.1, reduction='none')(logits.float(),labels.long()) 
            loss = torch.mean(loss)
        else:
            #import ipdb; ipdb.set_trace()
            loss = F.cross_entropy(logits.float(), labels.long())

        class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
        metrics = {'Class Loss': loss, 'Class Accuracy @ 1': class_acc1, 'Class Accuracy @ 5': class_acc5}

        return metrics


    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Class Loss']
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
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
