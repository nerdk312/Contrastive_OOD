import pytorch_lightning as pl
#import pytorch_lightning.metrics.functional as plm
import torch
import torch.nn.functional as F

from Contrastive_uncertainty.Moco.self_supervised import SSLEvaluator
from Contrastive_uncertainty.Moco.pl_metrics import accuracy

class SSLFineTuner(pl.LightningModule):

    def __init__(self, backbone, in_features, num_classes, hidden_dim=1024):
        """
        Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
        with 1024 units
        Example::
            from pl_bolts.utils.self_supervised import SSLFineTuner
            from pl_bolts.models.self_supervised import CPCV2
            from pl_bolts.datamodules import CIFAR10DataModule
            from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                        CPCTrainTransformsCIFAR10
            # pretrained model
            backbone = CPCV2.load_from_checkpoint(PATH, strict=False)
            # dataset + transforms
            dm = CIFAR10DataModule(data_dir='.')
            dm.train_transforms = CPCTrainTransformsCIFAR10()
            dm.val_transforms = CPCEvalTransformsCIFAR10()
            # finetuner
            finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)
            # train
            trainer = pl.Trainer()
            trainer.fit(finetuner, dm)
            # test
            trainer.test(datamodule=dm)
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.backbone = backbone # Nawid - encoder entwork
        self.ft_network = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=0.2,
            n_hidden=hidden_dim
        ) # Nawid - finetuning layer

    def on_train_epoch_start(self) -> None:
        self.backbone.eval() # Nawid - put backbone in evaluation mode

    # Nawid - calculates the loss
    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log('train_loss', loss.item(),on_epoch=True)
        self.log('train_acc',acc.item(),on_epoch = True)
        #self.logger.log_metrics(metrics, step=self.global_step)
        return loss

    # Nawid- calculates the validatiaon accuracy and the validation loss
    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log('val_loss', loss.item(),on_epoch=True)
        self.log('val_acc',acc.item(),on_epoch = True)
        
        #self.logger.log_metrics(metrics, step=self.global_step)
        return metrics

    # Nawid - Calculates the test information
    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log('test_loss', loss.item(),on_epoch=True)
        self.log('test_acc',acc.item(),on_epoch = True)
        #self.logger.log_metrics(metrics, step= self.global_step)

    # Nawid - step which is in common in training, validation and test
    def shared_step(self, batch):
        (x_q,x_k), y = batch

        with torch.no_grad():
            feats = self.backbone.encoder_q.representation_output(x_q) # Nawid - get the output representation of the data
        feats = feats.view(feats.size(0), -1)
        logits = self.ft_network(feats)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        #acc = plm.accuracy(logits, y)

        return loss, acc
    # Nawid- optimiser for the network
    def configure_optimizers(
        self,
    ):
        return torch.optim.Adam(self.ft_network.parameters(), lr=0.0002)