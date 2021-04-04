import torch
import torch.nn as nn
import torch.nn.functional as F

from Contrastive_uncertainty.toy_example.toy_encoder import Backbone
from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k
from Contrastive_uncertainty.toy_example.toy_module import Toy

class SoftmaxToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int =  20,
        emb_dim: int = 2,
        num_classes:int = 4,
        ):
        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        self.save_hyperparameters()
        #import ipdb;ipdb.set_trace()

        
        # Nawid - required to use for the fine tuning
        

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder= self.init_encoders()
        self.classifier = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

    # Instantiate classifier
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        return encoder
    
    def loss_function(self, batch, auxillary_data=None):
        
        (img_1, img_2), labels, indices = batch
        loss = self.forward(img_1, labels)
        return loss


    def forward(self, data,labels):
        z = self.encoder(data)
        z = F.relu(z)
        logits = self.classifier(z)
        loss = F.cross_entropy(logits, labels.long())
        acc1 = precision_at_k(logits, labels,)
    
        return loss 
    
    def feature_vector(self, data):
        z = self.encoder(data)
        return z
    
    '''
    def on_init(self, datamodule):
        return None
    '''
