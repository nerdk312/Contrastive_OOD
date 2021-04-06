import torch
import numpy as np

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
    loss = - 1 * mean_log_prob_pos
    #loss = - (model.hparams.softmax_temperature / model.hparams.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    
    return loss
    
def supervised_contrastive_loss(model, batch, auxillary_data=None):
    (img_1, img_2), labels = batch
    imgs = torch.cat([img_1, img_2], dim=0)
    bsz = labels.shape[0]
    features = model.encoder_q(imgs)
    ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
    loss = supervised_contrastive_forward(features, labels)
    return loss