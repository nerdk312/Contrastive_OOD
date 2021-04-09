import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.Moco.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy


def supervised_contrastive_forward(model, features, labels=None, mask=None):
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
        mask = torch.eye(batch_size, dtype=torch.float32, device = model.device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(model.device)
    else:
        mask = mask.float().to(model.device)
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
    anchor_count = contrast_count  # Nawid - all the different views are the anchors
    # compute logits
    anchor_dot_contrast = torch.div(  # Nawid - similarity between the anchor and the contrast feature
        torch.matmul(anchor_feature, contrast_feature.T),
        model.hparams.softmax_temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(model.device),
        0
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    #loss = - 1 * mean_log_prob_pos
    #loss = - (model.hparams.softmax_temperature / model.hparams.base_temperature) * mean_log_prob_pos
    loss = - (model.hparams.softmax_temperature / 0.07) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    
    return loss
    
def supervised_contrastive_loss(model, batch, auxillary_data=None):
    (img_1, img_2), labels = batch
    imgs = torch.cat([img_1, img_2], dim=0)
    bsz = labels.shape[0]
    features = model.encoder_q(imgs)
    features = nn.functional.normalize(features, dim=1)
    ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
    loss = supervised_contrastive_forward(model, features, labels)
    metrics = {'Supervised Contrastive Loss': loss}

    return metrics





def class_discrimination(model, x):
    """
    Input:
        x: a batch of images for classification
    Output:
        logits
    """
    # compute query features
    z = model.feature_vector(x) # Gets the feature map representations which I use for the purpose of pretraining
    z = F.relu(model.encoder_q.class_fc1(z))
    
    if model.hparams.normalize:
        z = nn.functional.normalize(z, dim=1)
    
    logits = model.encoder_q.class_fc2(z)
    return logits


def classification_loss(model, batch):
    (img_1, img_2), labels = batch
    logits = class_discrimination(model,img_1)
    if model.hparams.label_smoothing:
        loss = LabelSmoothingCrossEntropy(ε=0.1, reduction='none')(logits.float(),labels.long()) 
        loss = torch.mean(loss)
    else:
        loss = F.cross_entropy(logits.float(), labels.long())
    
    class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
    metrics = {'Class Loss': loss, 'Class Accuracy @ 1': class_acc1, 'Class Accuracy @ 5': class_acc5}

    return metrics


def moco_forward(model, img_q, img_k):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        logits, targets
    """
    # compute query features
    q = model.encoder_q(img_q)  # queries: NxC
    q = nn.functional.normalize(q, dim=1)
    # compute key features
    with torch.no_grad():  # no gradient to keys
        model._momentum_update_key_encoder()  # update the key encoder
        k = model.encoder_k(img_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # Nawid - dot product between query and queues
    # negative logits: NxK
    l_neg = torch.einsum('nc,ck->nk', [q, model.queue.clone().detach()])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
    # apply temperature
    logits /= model.hparams.softmax_temperature
    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long) # Nawid - class zero is always the correct class, which corresponds to the postiive examples in the logit tensor (due to the positives being concatenated first)
    labels = labels.type_as(logits)
    # dequeue and enqueue
    model._dequeue_and_enqueue(k)
    return logits, labels

def moco_loss(model, batch):
    (img_1, img_2), labels = batch
    output, target = moco_forward(model, img_q=img_1, img_k=img_2)
    loss = F.cross_entropy(output, target.long())
    acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
    metrics = {'Instance Loss': loss, 'Instance Accuracy @ 1': acc1, 'Instance Accuracy @ 5': acc5}

    return metrics


def uniform_loss(model,x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
def align_loss(model,x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
def class_align_loss(model,x,y,labels):
    class_alignment_loss = torch.tensor([0.0],device = model.device)
    full_data = torch.cat([x,y],dim=0) # concatenate the different augmented views
    full_labels = torch.cat([labels,labels],dim=0) # Double the labels to represent the labels for each view
    for i in range(model.hparams.num_classes):
        class_data= full_data[full_labels==i] # mask to only get features corresponding only the particular class
        class_dist = torch.pdist(class_data, p=2).pow(2).mean()
        class_alignment_loss += class_dist
    
    return class_alignment_loss

'''
        f1,f2 = self.feature_vector_compressed(img_1), self.feature_vector_compressed(img_2)
        align_loss = self.align_loss(f1,f2)
        uniformity_loss = (self.uniform_loss(f1) + self.uniform_loss(f2))/2
        loss = align_loss +uniformity_loss
        self.log('Training Alignment Loss', align_loss.item(),on_epoch=True)
        self.log('Training Uniformity Loss', uniformity_loss.item(),on_epoch=True)
        self.log('Training U+A Loss', loss.item(),on_epoch=True)
        '''
