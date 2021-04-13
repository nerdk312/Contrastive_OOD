import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from tqdm import tqdm
import faiss

from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean



class NNCLToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4, 
        hidden_dim: int = 20,
        emb_dim: int = 2,
        num_negatives: int = 32,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        num_cluster :list = [2],
        pretrained_network = None,
        ):
        """
        hidden_dim: dimensionality of neural network (default: 128)
        emb_dim: dimensionality of the feature space (default: 2)
        num_negatives: number of negative samples/prototypes (defaul: 16384)

        encoder_momentum: momentum for updating key encoder (default: 0.999)
        softmax_temperature: softmax temperature
        """

        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        
        self.save_hyperparameters()

        
        self.encoder = self.init_encoders()


        
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(self.hparams.hidden_dim,self.hparams.emb_dim)  
        return encoder


    @torch.no_grad()
    def compute_features(self, data):
        features = self.feature_vector(data)
        features = features.cpu().numpy() # numpy required for clustering
        return features


    def feature_vector(self, data):
        z = self.encoder(data)
        return z

    def cluster_data(self,data):
        features = self.compute_features(data)
        pseudo_labels = self.get_clusters(features, 10)
        pseudo_labels = torch.from_numpy(pseudo_labels).to(self.device) 
        return pseudo_labels
    # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
    # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device = self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        '''
        if self.hparams.contrast_mode == 'one':
            anchor_feature = features[:, 0] # Nawid - anchor is only the index itself and only the single view
            anchor_count = 1 # Nawid - only one anchor
        elif self.hparams.contrast_mode == 'all':
            anchor_feature = contrast_feature 
            anchor_count = contrast_count # Nawid - all the different views are the anchors
        else:
            raise ValueError('Unknown mode: {}'.format(self.hparams.contrast_mode))
        '''
        
        anchor_feature = contrast_feature 
        anchor_count = contrast_count # Nawid - all the different views are the anchors

        # compute logits
        anchor_dot_contrast = torch.div( # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.hparams.softmax_temperature / self.hparams.softmax_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

    # Assign clusters to the data
    def get_clusters(self,ftrain, nclusters):
        kmeans = faiss.Kmeans(
            ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=True
        )
        kmeans.train(np.random.permutation(ftrain))
        _, ypred = kmeans.assign(ftrain)
        return ypred

    def loss_function(self,batch,cluster_result=None):
        metrics = {}
        (img_1,img_2), labels,indices = batch
        pseudo_labels = self.cluster_data(img_1)
        imgs = torch.cat([img_1, img_2], dim=0)
        bsz = labels.shape[0]
        features = self.encoder(imgs)
        ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
        loss = self.forward(features, pseudo_labels)
        
        import ipdb; ipdb.set_trace()
        return metrics

    def loss_function(self, batch ,auxillary_data=None):
        (img_1, img_2), labels, indices = batch
        imgs = torch.cat([img_1, img_2], dim=0)
        bsz = labels.shape[0]
        features = self.encoder(imgs)
        ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
        loss = self.forward(features, labels)
        metrics = {'Loss':loss}

        return metrics