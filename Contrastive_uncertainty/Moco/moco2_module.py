import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms
from Contrastive_uncertainty.Moco.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.Moco.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy

from Contrastive_uncertainty.Moco.loss_functions import moco_loss, classification_loss, supervised_contrastive_loss


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
        data_dir: str = './',
        batch_size: int = 32,
        use_mlp: bool = False,
        num_workers: int = 8,
        num_channels:int = 3, # number of channels for the specific dataset
        z_dim:int = 512, # dimensionality of the output of the resnet 18 after the global average pooling layer
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
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.class_names = [v for k,v in class_dict.items()]
        self.save_hyperparameters()


        # use CIFAR-10 by default if no datamodule passed in
        if datamodule is None:
            datamodule = CIFAR10DataModule(data_dir,batch_size = self.hparams.batch_size)
            datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
            datamodule.val_transforms = Moco2EvalCIFAR10Transforms()
            datamodule.test_transforms = Moco2EvalCIFAR10Transforms()

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
        
        '''
        encoder_q = MocoEncoder(latent_size =self.hparams.emb_dim,feature_map_size = self.hparams.z_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
        encoder_k = MocoEncoder(latent_size =self.hparams.emb_dim,feature_map_size = self.hparams.z_dim,num_channels = self.hparams.num_channels,num_classes = self.hparams.num_classes)
        '''
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

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

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

    

    def training_step(self, batch, batch_idx):
        
        #(img_1, img_2), labels = batch
         
        loss = torch.tensor([0.0], device=self.device)
        if self.hparams.contrastive:
            metrics = moco_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
            loss += metrics['Instance Loss']
        
        if self.hparams.classifier:
            metrics = classification_loss(self,batch)
            for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
            loss += metrics['Class Loss']
        
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

        #return results
    '''
    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {
            'val_loss': val_loss,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5
        }
        return {'val_loss': val_loss, 'log': log, 'progress_bar': log}
    '''

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
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
            loss += metrics['Supervised Contrastive Loss']

        
        metrics = classification_loss(self,batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss += metrics['Class Loss']


        self.log('Test Total Loss', loss.item(),on_epoch=True)
        
        #return {'logits':logits,'target':labels} # returns y_pred as y_pred are essentially the logits in this case, and I want to log how the logits change in time
    '''
    def test_epoch_end(self, test_step_outputs):
        # Saves the predictions which are the logits in this case to see how the logits are changing in time
        
        #flattened_logits = torch.flatten(torch.cat([output['logits']for output in validation_step_outputs])) #  concatenate the logits
        #self.logger.experiment.log(
        #    {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
        #    "global_step": self.global_step})
        
        preds = torch.cat([output['logits'] for output in test_step_outputs])
        targets = torch.cat([output['target'] for output in test_step_outputs])

        top_pred_ids = preds.argmax(axis=1)

        self.logger.experiment.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(probs = preds.cpu().numpy(),
            preds=None, y_true=targets.cpu().numpy(),
            class_names=self.class_names),
            "global_step": self.global_step
                  })
    '''    


    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer
    '''
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder_q.load_state_dict(checkpoint['online_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])
    '''

    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder_q.load_state_dict(checkpoint['target_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])

    
    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output