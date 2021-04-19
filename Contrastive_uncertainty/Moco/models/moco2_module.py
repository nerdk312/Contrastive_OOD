import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.Moco.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.Moco.models.loss_functions import moco_loss, classification_loss, supervised_contrastive_loss


class MocoV2(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        batch_size: int = 32,
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
        num_classes:int = 10, # Attribute required for the finetuning value
        classifier: bool = False,
        contrastive: bool = True,
        supervised_contrastive: bool = False,
        normalize:bool = True,
        class_dict:dict = None,
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        label_smoothing:bool = False,
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.num_classes = num_classes
        self.class_names = [v for k,v in class_dict.items()]
        self.save_hyperparameters()

        self.datamodule = datamodule

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()
        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            # Double checking if the training of the model was done correctly
            #param_q.requires_grad = False
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder_q = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
            encoder_k = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder_q = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
            encoder_k = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
        
        
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

    '''
    def feature_vector(self,x):
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder_q.representation_output(x)
        z = nn.functional.normalize(z, dim=1)
        return z
    
    def feature_vector_compressed(self,x):
        """
        Input:
            x: a batch of images for classification
        Output:
            logits
        """
        # compute query features
        z = self.feature_vector(x) # Gets the feature map representations which I use for the purpose of pretraining
        z = self.encoder_q.class_fc1(z)
        z = nn.functional.normalize(z, dim=1)
        return z
    '''    
    def callback_vector(self,x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder_k(x)
        z = nn.functional.normalize(z, dim=1)
        return z

    def training_step(self, batch, batch_idx):
        loss = torch.tensor([0.0], device=self.device)
        if self.hparams.classifier:
            metrics = classification_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
            loss += metrics['Class Loss']

        if self.hparams.contrastive:
            metrics = moco_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
            loss += metrics['Instance Loss']
        
        if self.hparams.supervised_contrastive:
            metrics = supervised_contrastive_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
            loss += metrics['Supervised Contrastive Loss']
        
        self.log('Training Total Loss', loss.item(),on_epoch=True)

        return loss
        

    def validation_step(self, batch, batch_idx,dataset_idx):
        loss = torch.tensor([0.0], device=self.device)
        if self.hparams.contrastive:
            metrics = moco_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
            loss += metrics['Instance Loss']
        
        if self.hparams.classifier:
            metrics = classification_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
            loss += metrics['Class Loss']
        
        if self.hparams.supervised_contrastive:
            metrics = supervised_contrastive_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
            loss += metrics['Supervised Contrastive Loss']
        
        self.log('Validation Total Loss', loss.item(),on_epoch=True)


    def test_step(self, batch, batch_idx):
        loss = torch.tensor([0.0], device=self.device)
        if self.hparams.contrastive:
            metrics = moco_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
            loss += metrics['Instance Loss']

        if self.hparams.supervised_contrastive:
            metrics = supervised_contrastive_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
            loss += metrics['Supervised Contrastive Loss']

        if self.hparams.PCL:
            metrics = pcl_loss(self,batch,self.auxillary_data)
            for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
            loss += metrics['PCL Loss']
        
        metrics = classification_loss(self,batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        loss += metrics['Class Loss']


        self.log('Test Total Loss', loss.item(),on_epoch=True)
        
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
        self.encoder_q.load_state_dict(checkpoint['target_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])
