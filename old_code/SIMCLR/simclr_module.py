import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from warnings import warn
from torchvision.models import densenet

from datamodules.cifar10_datamodule import CIFAR10DataModule
from datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from datamodules.mnist_datamodule import MNISTDataModule
from datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms
from Moco.resnet_models import resnet18, resnet34,resnet50
from Moco.evaluator import Flatten
from SIMCLR.simclr_loss import nt_xent_loss

from optimizers.lars_scheduling import LARSWrapper
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR




class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

# Module to perform classification , with option to normalize the feature before the logits
class Classifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, num_classes=10,normalize = False): 
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes, bias=False))

    def forward(self, x):
        x = self.model[:-1](x)
        if self.normalize:
            x = F.normalize(x,dim=1)
        x = self.model[-1](x)
        return x


class SimCLR(pl.LightningModule):
    def __init__(self,
                batch_size,
                num_samples,
                datamodule = None,
                num_channels = 3,
                warmup_epochs=10,
                lr=1e-4,
                opt_weight_decay=1e-6,
                loss_temperature=0.5,
                **kwargs):
    
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()


        # h -> || -> z
        self.projection = Projection()

        data_dir = './',
        # use CIFAR-10 by default if no datamodule passed in
        if datamodule is None:
            datamodule = CIFAR10DataModule(data_dir,batch_size = self.hparams.batch_size)
            datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
            datamodule.val_transforms = Moco2EvalCIFAR10Transforms()
            datamodule.test_transforms = Moco2EvalCIFAR10Transforms()

        self.datamodule = datamodule

    def init_encoder(self):# Need to change
        encoder = resnet18()

        # when using cifar10, replace the first conv so image doesn't shrink away
        encoder.conv1 = nn.Conv2d(
            self.hparams.num_channels, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        return encoder
    
    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]
    
    def setup(self, stage): # Likely do not need this 
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size
    
    
    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.lr))

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]
    
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('Training Instance Loss', loss.item(),on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx,dataset_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('Validataion Instance Loss', loss.item(),on_epoch=True)
    
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('Test Instance Loss', loss.item(),on_epoch=True)

        
    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        import ipdb; ipdb.set_trace()
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss
        