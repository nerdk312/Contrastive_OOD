import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k
from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

class MocoToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int = 20,
        emb_dim: int = 2,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        pretrained_network = None,
        num_classes:int = 4,
        ):
        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()
        self.classifier = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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


    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        #import ipdb; ipdb.set_trace()
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # Nawid - dot product between query and queues
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long) # Nawid - class zero is always the correct class, which corresponds to the postiive examples in the logit tensor (due to the positives being concatenated first)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
    
    def loss_function(self, batch, auxillary_data=None):
        (img_1, img_2), labels, indices = batch
         
        output, target = self(img_q=img_1, img_k=img_2)
        
        loss = F.cross_entropy(output, target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        metrics = {'Loss': loss, 'Accuracy @ 1': acc1, 'Accuracy @5': acc5}
        return metrics

    def feature_vector(self, data):
        z = self.encoder_q(data)
        return z

    # Loads both network as a target state dict
    def encoder_loading(self, pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder_q.load_state_dict(checkpoint['target_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])
    
    def class_discrimination(self, data):
        z = self.encoder_q(data)
        z = F.relu(z)
        logits = self.classifier(z)
        return logits

    '''
    def on_train_epoch_start(self, datamodule):
        return None
    '''