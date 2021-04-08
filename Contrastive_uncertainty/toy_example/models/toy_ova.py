import torch
import torch.nn as nn
import torch.nn.functional as F

from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

class OVAToy(Toy):
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
        one_hot_labels = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        centroids = self.update_embeddings(img_1, labels)
        y_pred = self(img_1, centroids)
        loss = F.binary_cross_entropy(y_pred, one_hot_labels)
        acc1, = precision_at_k(y_pred, labels)
        '''
        correct = torch.argmax(y_pred[:original_xs_length].detach(),dim=1).view(original_xs_length,-1) == labels # look at calculating the correct values only for the case of the true data
        accuracy = torch.mean(correct.float())
        '''

        metrics = {'Loss': loss, 'Accuracy @ 1': acc1}
        return metrics

    def feature_vector(self, x): # Obtain feature vector
        x = self.encoder(x)
        x = nn.functional.normalize(x, dim=1)
        return x

    def forward(self, x, centroids): # obtain predictions
        z = self.feature_vector(x)
        distances = self.euclidean_dist(z, centroids)
        
        y_pred = 2*torch.sigmoid(distances)
        return y_pred  # shape (batch,num_classes)



    def class_discrimination(self, x, centroids): # same as forward
        y_pred = self(x,centroids)
    
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