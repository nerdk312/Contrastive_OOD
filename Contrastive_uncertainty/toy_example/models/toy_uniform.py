import torch
import torch.nn as nn
import torch.nn.functional as F

from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

class UniformityToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int =  20,
        emb_dim: int = 2,
        num_classes:int = 2,
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
        z =  self.feature_vector(img_1)
        loss = self.uniform_loss(z)
        metrics = {'Loss': loss}
        return metrics

    # Uniformity and alignment
    def uniform_loss(self,x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def feature_vector(self, data):
        z = self.encoder(data)
        return z
    
    def class_discrimination(self, data):
        z = self.encoder(data)
        z = F.relu(z)
        logits = self.classifier(z)
        return logits
    
    @torch.no_grad()
    def update_embeddings(self, x, labels): # Assume y is one hot encoder
        z = self.feature_vector(x)  # (batch,features)
        y = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        # compute sum of embeddings on class by class basis

        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        features_sum = torch.matmul(y.T,z) # (num_classes,batch) (batch,features) to get (num_class, features)

        #features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
        

        embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings

    def euclidean_dist(self, x, y):  # Calculates the difference
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        diff = x - y
        distances = -torch.pow(diff, 2).sum(2)  # Need to get the negative distance
        return distances
    
    def centroid_confidence(self, x,centroids):
        z = self.feature_vector(x)
        distances = self.euclidean_dist(z, centroids)
    
        return distances  # shape: (batch, num classes)
        