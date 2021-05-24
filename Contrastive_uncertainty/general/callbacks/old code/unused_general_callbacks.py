import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import sklearn.metrics as skm

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix

class Uniformity(pl.Callback):
    def __init__(self, t,datamodule,quick_callback):
        super().__init__()
        self.t  = t
        self.datamodule = datamodule
        self.quick_callback = quick_callback
        self.log_name = 'uniformity'
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)
    '''
    
    def on_validation_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)

    def obtain_features(self,pl_module):
        features = []
        dataloader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, label,indices) in enumerate(loader):
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            img = img.to(pl_module.device)
            features.append(pl_module.callback_vector(img))
        
        features = torch.cat(features) # Obtaint the features for the representation
        return features

    def calculate_uniformity(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features) # Change to a torch tensor

        uniformity = torch.pdist(features, p=2).pow(2).mul(-self.t).exp().mean().log()
        wandb.log({self.log_name:uniformity.item()})
        #uniformity = uniformity.item()
        return uniformity


class Centroid_distance(pl.Callback):
    def __init__(self,datamodule,quick_callback):
        super().__init__()
        self.datamodule = datamodule
        self.quick_callback = quick_callback
        self.log_distance_name = 'centroid_distance'
        self.log_rbf_similarity_name = 'centroid_rbf_similarity'
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        optimal_centroids = self.optimal_centroids(pl_module)
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        collated_distances = []
        collated_rbf_similarity = []
        for step, (data,labels) in enumerate(loader):
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
                data, labels =  data.to(pl_module.device), labels.to(pl_module.device)
                latent_vector = pl_module.feature_vector(data)

            distances = self.distance(pl_module,latent_vector,optimal_centroids)
            #import ipdb;ipdb.set_trace()
            labels = labels.reshape(len(labels),1)
            # gather takes the values from distances along dimension 1 based on the values of labels
            class_distances = torch.gather(distances,1,labels)
            class_rbf_sim = self.rbf_similarity(class_distances)

            collated_distances.append(class_distances)
            collated_rbf_similarity.append(class_rbf_sim)

        collated_distances = torch.cat(collated_distances) # combine all the values
        collated_rbf_similarity = torch.cat(collated_rbf_similarity)

        mean_distance = torch.mean(collated_distances)
        mean_rbf_similarity = torch.mean(collated_rbf_similarity)

        wandb.log({self.log_distance_name: mean_distance.item()})
        wandb.log({self.log_rbf_similarity_name:mean_rbf_similarity})
    '''

    def on_validation_epoch_end(self,trainer,pl_module):
        optimal_centroids = self.optimal_centroids(pl_module)
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        collated_distances = []
        collated_rbf_similarity = []
        for step, (data,labels) in enumerate(loader):
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
                data, labels =  data.to(pl_module.device), labels.to(pl_module.device)
                latent_vector = pl_module.callback_vector(data)

            distances = self.distance(pl_module,latent_vector,optimal_centroids)
            #import ipdb;ipdb.set_trace()
            labels = labels.reshape(len(labels),1)
            # gather takes the values from distances along dimension 1 based on the values of labels
            class_distances = torch.gather(distances,1,labels)
            class_rbf_sim = self.rbf_similarity(class_distances)

            collated_distances.append(class_distances)
            collated_rbf_similarity.append(class_rbf_sim)

        collated_distances = torch.cat(collated_distances) # combine all the values
        collated_rbf_similarity = torch.cat(collated_rbf_similarity)

        mean_distance = torch.mean(collated_distances)
        mean_rbf_similarity = torch.mean(collated_rbf_similarity)

        wandb.log({self.log_distance_name: mean_distance.item()})
        wandb.log({self.log_rbf_similarity_name:mean_rbf_similarity})

        #import ipdb; ipdb.set_trace()
        

    def optimal_centroids(self,pl_module):
        centroids_list = []
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        for step, (data,labels) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
            data,labels = data.to(pl_module.device), labels.to(pl_module.device)

            centroids = self.update_embeddings(pl_module, data, labels)
            centroids_list.append(centroids)
        
        collated_centroids = torch.stack(centroids_list)
        optimal_centroids = torch.mean(collated_centroids, dim=0)
        return optimal_centroids

    @torch.no_grad()
    def update_embeddings(self, pl_module, x, labels): # Assume y is one hot encoder
        z = pl_module.feature_vector(x) # (batch,features)
        y = F.one_hot(labels,num_classes = pl_module.num_classes).float()
        # compute sum of embeddings on class by class basis

        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        #features_sum = torch.matmul(y.T,z) # (num_classes,batch) (batch,features) to get (num_class, features)

        features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
        embeddings = features_sum / y.sum(0)
        #embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes)
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings

    def distance(self,pl_module, x, centroids):
        n = x.size(0) # (batch,features)
        m = centroids.size(0) # (num classes, features)
        d = x.size(1)
        
        assert d == centroids.size(1)

        x = x.unsqueeze(1).expand(n, m, d) # (batch,num_classes, features)
        centroids = centroids.unsqueeze(0).expand(n, m, d) # (batch,num_classes,features)
        diff = x - centroids # (batch,num_classes,features) 
        distances = diff.sum(2) # (batch,num_classes) - distances to each class centroid for each data point in the batch
        #distances = -torch.pow(diff,2).sum(2) # Need to get the negative distance
        return distances

    def rbf_similarity(self,distances):
        exp_similarity = (-(distances**2)).div(2*1**2).exp() # square the distances and divide by a scaling temr
        return exp_similarity
    

        #confidence, indices = torch.max(y_pred,dim=1)
        
    #\n",    
    "        confidence,indices =  confidence.reshape(len(confidence),1), indices.reshape(len(indices),1) # reshape the tensors\n",
    "        density_targets = torch.zeros(len(confidence),2).to(self.device)\n",
    "        density_targets.scatter(1,indices,confidence) # place the values of confidences in the locations specified by indices\n",

class SupConLoss(pl.Callback):
    def __init__(self,datamodule,quick_callback, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.datamodule = datamodule
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.quick_callback = quick_callback
        self.log_name = 'SupCon' 
    
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        dataloader  = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, dataloader)
        for i, (imgs, labels) in enumerate(loader):
            # Obtain the image representations of the data for the specific task
            # Concatenate the two different views of the same images
            imgs = torch.cat([imgs[0],imgs[1]],dim=0) 
            bsz = labels.shape[0]

            imgs,labels = imgs.to(pl_module.device), labels.to(pl_module.device)

            features = pl_module.feature_vector(imgs)
            features_q, features_k = torch.split(features, [bsz,bsz],dim=0)
            features = torch.cat([features_q.unsqueeze(1),features_k.unsqueeze(1)],dim=1)
            self.forward(pl_module,features,labels)
    '''

    # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
    # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
    def forward(self,pl_module, features, labels=None, mask=None):
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
            mask = torch.eye(batch_size, dtype=torch.float32, device = pl_module.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(pl_module.device)
        else:
            mask = mask.float().to(pl_module.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # Nawid - anchor is only the index itself and only the single view
            anchor_count = 1 # Nawid - only one anchor
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature 
            anchor_count = contrast_count # Nawid - all the different views are the anchors
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div( # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(pl_module.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        wandb.log({self.log_name: loss.item()})

        return loss