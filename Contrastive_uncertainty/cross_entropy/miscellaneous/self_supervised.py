import math

import pytorch_lightning as pl
import torch
#from pytorch_lightning.metrics.functional import accuracy

from torch.nn import functional as F

from Contrastive_uncertainty.Moco.evaluator import SSLEvaluator
from Contrastive_uncertainty.Moco.pl_metrics import accuracy


class SSLOnlineEvaluator(pl.Callback):  # pragma: no-cover

    def __init__(self, drop_p: float = 0.2, hidden_dim: int = 1024, z_dim: int = None, num_classes: int = None):
        """
        Attaches a MLP for finetuning using the standard self-supervised protocol.
        Example::
            from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
            # your model must have 2 attributes
            model = Model()
            model.z_dim = ... # the representation dim
            model.num_classes = ... # the num of classes in the model
        Args:
            drop_p: (0.2) dropout probability
            hidden_dim: (1024) the hidden dimension for the finetune MLP
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None
        self.z_dim = z_dim
        self.num_classes = num_classes

    def on_pretrain_routine_start(self, trainer, pl_module):
        

        # attach the evaluator to the module

        if hasattr(pl_module, 'z_dim'):
            self.z_dim = pl_module.z_dim
        if hasattr(pl_module, 'num_classes'):
            self.num_classes = pl_module.num_classes

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim
        ).to(pl_module.device)

        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x):
        """
        Override this to customize for the particular model
        Args:
            pl_module:
            x:
        """
        if len(x) == 2 and isinstance(x, list):
            x = x[0]

        representations = pl_module.encoder_q.representation_output(x) # Nawid - Made a custom funciton to get the representation of the encoder directly
        representations = representations.reshape(representations.size(0), -1)
        return representations
    ''' Nawid - for the case where there is only a single output from the data augmentation
    def to_device(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        return x, y
    '''
    # Nawid - for the case where there  are two outputs from the data augmentation
    def to_device(self,batch, device):
        #import ipdb; ipdb.set_trace()
        (x_q,x_k), (y) = batch
        x_q = x_q.to(device)
        y = y.to(device)
        return x_q, y


    def on_train_batch_end(self, trainer, pl_module,outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)
        #import ipdb; ipdb.set_trace()
        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        acc = accuracy(mlp_preds, y) 
        '''
        if trainer.datamodule is not None:
            acc = accuracy(mlp_preds, y, num_classes=self.num_classes)
        else:
            acc = accuracy(mlp_preds, y)
        '''

        metrics = {'ft_callback_mlp_loss': mlp_loss, 'ft_callback_mlp_acc': acc}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)
