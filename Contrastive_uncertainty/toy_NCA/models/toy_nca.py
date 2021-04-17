import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import math

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k

from Contrastive_uncertainty.toy_NCA.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_NCA.models.toy_module import Toy
from Contrastive_uncertainty.toy_NCA.models.LinearAverage import LinearAverage
from Contrastive_uncertainty.toy_NCA.models.NCA import NCACrossEntropy

class NCAToy(Toy):
    def __init__(self,
        datamodule,
        labels,
        margin :int = 0,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int = 20,
        emb_dim: int = 2,
        num_classes:int = 2,
        temperature:float: = 0.07,
        memory_momentum: float = 0.05
        ):

        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        

        self.lemniscate = LinearAverage(emb_dim, ndata, temperature, memory_momentum).cuda()
        self.criterion = NCACrossEntropy(torch.LongTensor([y for (p, y) in train_loader.dataset.imgs]),
            margin / temperature).cuda()

        
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()
        self.classifier = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

    def forward(self, x, indexes):
        batchSize = x.size(0)
        # Nawid - n is the dimensionality I believe
        n = x.size(1)
        exp = torch.exp(x)

        # labels for currect batch
        # Nawid - obtain the labels for the specific data points
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        # Nawid -  repeat the labels
        same = y.repeat(1, n).eq_(self.labels)

        # self prob exclusion, hack with memory for effeciency
       # Nawid -  prevent having the nearest neighbour of the data point as itself
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        # Nawid - obtain the probability of a data point
        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize



    '''
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        encoder_k = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        
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
        assert self.hparams.num_negatives % batch_size == 0, "Likely using a small batch due to not dropping last"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # Nawid - add the keys to the queue
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr
    '''
    
    def loss_function(self, batch, auxillary_data=None):
        data, labels, indices = batch


        output, target = self(img_q=img_1, img_k=img_2)
        
        loss = F.cross_entropy(output, target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        metrics = {'Loss': loss, 'Accuracy @ 1': acc1, 'Accuracy @5': acc5}
        return metrics     