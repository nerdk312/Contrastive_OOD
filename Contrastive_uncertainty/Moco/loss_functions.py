import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.Moco.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy

# Supervised contrastive loss
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
    (img_1, img_2), labels,indices = batch
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


def classification_loss(model, batch,auxillary_data = None):
    (img_1, img_2), labels,indices = batch
    logits = class_discrimination(model,img_1)
    if model.hparams.label_smoothing:
        loss = LabelSmoothingCrossEntropy(ε=0.1, reduction='none')(logits.float(),labels.long()) 
        loss = torch.mean(loss)
    else:
        loss = F.cross_entropy(logits.float(), labels.long())
    
    class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
    metrics = {'Class Loss': loss, 'Class Accuracy @ 1': class_acc1, 'Class Accuracy @ 5': class_acc5}

    return metrics

# MOCO
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

def moco_loss(model, batch, auxillary_data= None):
    (img_1, img_2), labels,indices = batch
    output, target = moco_forward(model, img_q=img_1, img_k=img_2)
    loss = F.cross_entropy(output, target.long())
    acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
    metrics = {'Instance Loss': loss, 'Instance Accuracy @ 1': acc1, 'Instance Accuracy @ 5': acc5}

    return metrics

# PCL

def PCL_forward(model, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
        is_eval: return momentum embeddings (used for clustering)
        cluster_result: cluster assignments, centroids, and density
        index: indices for training samples
    Output:
        logits, targets, proto_logits, proto_targets
    """
    #import ipdb; ipdb.set_trace()
    if is_eval: # Nawid - obtain key outputs
        k = model.encoder_k(im_q)
        print('k')
        k = nn.functional.normalize(k, dim=1) # Nawid - return normalised momentum embedding
        return k
    # compute key features
    with torch.no_grad():  # no gradient to keys
        model._momentum_update_key_encoder()  # update the key encoder
        k = model.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1) # Nawid - normalised key embeddings
        
    # compute query features
    q = model.encoder_q(im_q)  # queries: NxC
    q = nn.functional.normalize(q, dim=1) # Nawid - normalised query embeddings
    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #Nawid - positive logit between output of key and query
    # negative logits: Nxr
    l_neg = torch.einsum('nc,ck->nk', [q, model.queue.clone().detach()]) # Nawid - negative logits (dot product between key and negative samples in a query bank)
    # logits: Nx(1+r)
    logits = torch.cat([l_pos, l_neg], dim=1) # Nawid - total logits - instance based loss to keep property of local smoothness
    # apply temperature
    logits /= model.hparams.softmax_temperature
    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long,device = model.device)
    #labels = labels.type_as(logits)
    # dequeue and enqueue
    model._dequeue_and_enqueue(k) # Nawid - queue values
     
    # prototypical contrast - Nawid - performs the protoNCE
    if cluster_result is not None:
        proto_labels = []
        proto_logits = []
        for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
            #import ipdb; ipdb.set_trace()
            # get positive prototypes
            pos_proto_id = im2cluster[index] # Nawid - get the true cluster assignment for each of the different samples
            pos_prototypes = prototypes[pos_proto_id] # Nawid- prototypes is a kxd array of k , d dimensional clusters. Therefore this chooses the true clusters for the positive samples. Therefore this is a [B x d] matrix
            # sample negative prototypes
            all_proto_id = [i for i in range(im2cluster.max())] # Nawid - obtains all the cluster ids which were present
            neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist()) # Nawid - all the negative clusters are the set of all prototypes minus the set of all the negative prototypes
            neg_proto_id = sample(neg_proto_id, model.hparams.num_negatives) #sample r negative prototypes
            #neg_proto_id = neg_proto_id.to(self.device)
            neg_prototypes = prototypes[neg_proto_id] # Nawid - sample negative prototypes
            proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0) # Nawid - concatenate positive and negative prototypes, so this is  a [bxd] concatenated with [rxd] to make a [b + r xd]
            # compute prototypical logits
            logits_proto = torch.mm(q,proto_selected.t().to(model.device)) # Nawid - dot product between query and the prototypes (where the selected prototypes are transposed). The matrix multiplication is  [b x d] . [dx b +r] to make a [b x b +r]
            # targets for prototype assignment
            labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0),device = model.device).long()# Nawid - targets for the prototypes, this is a 1D vector with values from 0 to q-1 which represents that the value which shows that the diagonal should be the largest value
            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).to(model.device)], dim=0)]
            logits_proto /= temp_proto
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)
        return logits, labels, proto_logits, proto_labels
    else:
        return logits, labels, None, None

@torch.no_grad()
def compute_features(model, dataloader):
    print('Computing features ...')
    #import ipdb;ipdb.set_trace()
    
    features = torch.zeros(len(dataloader.dataset), model.hparams.emb_dim, device = model.device)
    for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
        assert len(dataloader)>0, 'Empty dataloader'
        assert max(indices) < len(dataloader.dataset),'indices higher than indices in dataset' # Test for the code to work
        if isinstance(images, tuple) or isinstance(images, list):
            images, *aug_images = images
            #import ipdb; ipdb.set_trace()
        images = images.to(model.device)
        feat = PCL_forward(model, images, is_eval=True)   # Nawid - obtain momentum features
        features[indices] = feat  # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
    return features.cpu()



def run_kmeans(model,x):
    """
    Args:
        x: data to be clustered
    """
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]} # Nawid -k-means results placed here
    for seed, num_cluster in enumerate(model.hparams.num_cluster): # Nawid - k-means clustering is performed several times for different values of k (according to the paper)
        # intialize faiss clustering parameters
        d = x.shape[1] # Nawid - dimensionality of the vector
        k = int(num_cluster) # Nawid - num cluster
        clus = faiss.Clustering(d, k) # Nawid -cluster object
        clus.verbose = True # Nawid - getter for the verbose property of clustering
        clus.niter = 20 # Nawid - getter for the number of k mean iterations
        clus.nredo = 5 # Nawid - getter for nredo property
        clus.seed = seed # Nawid - getter for seed property of clustering
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0 # gpu device number zero
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I] # Nawid - places cluster assignments for each sample  (image sample is assigned to a cluster)
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d) # Nawid - obtain the cluster centroids for the k clusters which are d dimensional
        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)] # Nawid - k different lists for the k clusters
        for im,i in enumerate(im2cluster): # Nawid - i is the cluster assignment
            Dcluster[i].append(D[im][0]) # Nawid - place in ith list (corresponding to ith cluster), the distance of the training example to the ith cluster (I think)
        # concentration estimation (phi)
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d
        #if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax
        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = model.hparams.softmax_temperature*density/density.mean()  #scale the mean to temperature
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(model.device)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(im2cluster).to(model.device)
        density = torch.Tensor(density).to(model.device)
        results['centroids'].append(centroids) # Nawid - (k,d) matrix which corresponds to k different d-dimensional centroids
        results['density'].append(density) # Nawid - concentation
        results['im2cluster'].append(im2cluster) # Nawid - list of the what image each particular cluster is present in
    return results


def cluster_data(model,dataloader):
    features = compute_features(model,dataloader)
    # placeholder for clustering result
    cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
    for num_cluster in model.hparams.num_cluster: # Nawid -Makes separate list for each different k value of the cluster (clustering is performed several times with different values of k), array of zeros for the im2cluster, the centroids and the density/concentration
        cluster_result['im2cluster'].append(torch.zeros(len(dataloader.dataset),dtype=torch.long,device = model.device))
        cluster_result['centroids'].append(torch.zeros(int(num_cluster),model.hparams.emb_dim,device = model.device))
        cluster_result['density'].append(torch.zeros(int(num_cluster),device = model.device))
    
     #if using a single gpuif args.gpu == 0:
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
    features = features.numpy()
    # Nawid - compute K-means
    cluster_result = run_kmeans(features)  #run kmeans clustering on master node
    return cluster_result

def pcl_loss(model,batch,cluster_result=None):
    metrics = {}
    (img_1,img_2), labels,indices = batch
    # compute output -  Nawid - obtain instance features and targets as  well as the information for the case of the proto loss
    output, target, output_proto, target_proto = PCL_forward(im_q=img_1, im_k=img_2, cluster_result=cluster_result, index=indices) # Nawid- obtain output
    # InfoNCE loss
    loss = F.cross_entropy(output, target) # Nawid - instance based info NCE loss
    acc_i = precision_at_k(output, target)[0]
    instance_metrics = {'Instance Loss':loss,'Instance Accuracy @ 1': acc_i}
    metrics.update(instance_metrics)
    # ProtoNCE loss
    if output_proto is not None:
        loss_proto = 0
        for index, (proto_out,proto_target) in enumerate(zip(output_proto, target_proto)): # Nawid - I believe this goes through the results of the m different k clustering results
            loss_proto += F.cross_entropy(proto_out, proto_target) #
            accp = precision_at_k(proto_out, proto_target)[0]
           # acc_proto.update(accp[0], images[0].size(0))
            # Log accuracy for the specific case
            proto_metrics = {'Accuracy @ 1 Cluster '+str(model.hparams.num_cluster[index]): accp}
            metrics.update(proto_metrics)
        # average loss across all sets of prototypes
        loss_proto /= len(model.hparams.num_cluster) # Nawid -average loss across all the m different k nearest neighbours
        loss += loss_proto # Nawid - increase the loss
    additional_metrics = {'PCL Loss':loss, 'ProtoLoss':loss_proto}
    metrics.update(additional_metrics)
    return metrics

def aux_data(model, dataloader):
    if model.hparams.PCL: # Boolean to choose wheter to perform clustering or not
        cluster_result = cluster_data(model,dataloader)
    else:
        cluster_result = None
    return cluster_result


# Uniformity and alignment
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
