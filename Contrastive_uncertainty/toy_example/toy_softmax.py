import torch
import torch.nn as nn
import torch.nn.functional as F

from Contrastive_uncertainty.toy_example.toy_encoder import Backbone
from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k


class SoftmaxToy(nn.Module):
    def __init__(self,
        hidden_dim: int =  20,
        emb_dim: int = 2,
        num_classes:int = 4,
        
        ):
        super().__init__()
        # Nawid - required to use for the fine tuning
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder= self.init_encoders()
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    # Instantiate classifier
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(self.hidden_dim,self.emb_dim)
        return encoder
    
    def loss_function(self, batch, auxillary_data=None):
        
        (img_1, img_2), labels = batch
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
    
    def on_train_epoch_start(self, datamodule):
        return None
