import numpy as np
from random import sample
import wandb

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from tqdm import tqdm
import faiss


from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.Moco.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50

from Contrastive_uncertainty.PCL.callbacks.general_callbacks import quickloading



class base_module(pl.LightningModule):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes:int = 10,
        class_dict:dict = None):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.num_classes = num_classes
        self.class_names = [v for k,v in class_dict.items()]
        self.datamodule = datamodule # Used for the purpose of obtaining data loader for the case of epoch starting
        
        
    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        #import ipdb; ipdb.set_trace()
        for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch, self.auxillary_data)
        for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)

    def loss_function(self, batch, auxillary_data=None):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer




class SupConPCLModule(base_module):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes:int = 10,
        class_dict:dict = None,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        num_cluster: list = [100],
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
        classifier: bool = False,
        normalize:bool = True,
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None):


        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay,num_classes,class_dict)
        
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            # Double checking if the training of the model was done correctly
            #param_q.requires_grad = False
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # ptr shows the position in the queue for the data
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Added index to show the label of the value in the queue, several dimensions to represent the value it is placed in all the clusters
        self.register_buffer("queue_index", torch.zeros(len(self.hparams.num_cluster),num_negatives, dtype=torch.long))
        
        
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
    def _dequeue_and_enqueue(self, keys,key_indices):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity , num negatives is divisible by batch size

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # Nawid - add the keys to the queue
        
        # Place the cluster assignments for the particular samples, # pos values im2cluster[index] gives [batch,1], which I could concatenate to get [batch, self.hparams.num_cluster] then concatenate to get num cluster 
        self.queue_index[:, ptr:ptr + batch_size] = key_indices.T

        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self,img_1,img_2,pseudo_labels):
         # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)


        
        


        
        

    '''

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
        
        #if self.hparams.contrast_mode == 'one':
        #    anchor_feature = features[:, 0] # Nawid - anchor is only the index itself and only the single view
        #    anchor_count = 1 # Nawid - only one anchor
        #elif self.hparams.contrast_mode == 'all':
        #    anchor_feature = contrast_feature 
        #    anchor_count = contrast_count # Nawid - all the different views are the anchors
        #else:
        #    raise ValueError('Unknown mode: {}'.format(self.hparams.contrast_mode))
        #
        
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
    '''
    '''
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
    '''
    

    
    def feature_vector(self,x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder_q.representation_output(x)
        #z = nn.functional.normalize(z, dim=1)
        return z
    
    


    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device = self.device)
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder_k(images)  # Nawid - obtain features for the task
            feat = nn.normalize(feat, dim=1) # Obtain 12 normalised features for clustering
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()

    def run_kmeans(self,x):
        """
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]} # Nawid -k-means results placed here

        for seed, num_cluster in enumerate(self.hparams.num_cluster): # Nawid - k-means clustering is performed several times for different values of k (according to the paper)
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
            density = self.hparams.softmax_temperature*density/density.mean()  #scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).to(self.device)
            centroids = nn.functional.normalize(centroids, p=2, dim=1)

            im2cluster = torch.LongTensor(im2cluster).to(self.device)
            density = torch.Tensor(density).to(self.device)

            results['centroids'].append(centroids) # Nawid - (k,d) matrix which corresponds to k different d-dimensional centroids
            results['density'].append(density) # Nawid - concentation
            results['im2cluster'].append(im2cluster) # Nawid - list of the what image each particular cluster is present in

        return results

    def cluster_data(self,dataloader):
        features = self.compute_features(dataloader)
        # placeholder for clustering result
        cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
        for num_cluster in self.hparams.num_cluster: # Nawid -Makes separate list for each different k value of the cluster (clustering is performed several times with different values of k), array of zeros for the im2cluster, the centroids and the density/concentration
            cluster_result['im2cluster'].append(torch.zeros(len(dataloader.dataset),dtype=torch.long,device = self.device))
            cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.hparams.emb_dim,device = self.device))
            cluster_result['density'].append(torch.zeros(int(num_cluster),device = self.device))
        
         #if using a single gpuif args.gpu == 0:
        features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
        features = features.numpy()
        # Nawid - compute K-means
        cluster_result = self.run_kmeans(features)  #run kmeans clustering on master node
        return cluster_result

    
    def loss_function(self,batch,cluster_result=None):
        (img_1,img_2), labels,indices = batch




        # compute output -  Nawid - obtain instance features and targets as  well as the information for the case of the proto loss
        output, target, output_proto, target_proto = self(im_q=img_1, im_k=img_2, cluster_result=cluster_result, index=indices) # Nawid- obtain output

        # InfoNCE loss
        loss = F.cross_entropy(output, target) # Nawid - instance based info NCE loss

        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for index, (proto_out,proto_target) in enumerate(zip(output_proto, target_proto)): # Nawid - I believe this goes through the results of the m different k clustering results
                loss_proto += F.cross_entropy(proto_out, proto_target) #
                accp = precision_at_k(proto_out, proto_target)[0]
               # acc_proto.update(accp[0], images[0].size(0))
                # Log accuracy for the specific case
                proto_metrics = {'Accuracy @ 1 Cluster '+str(self.hparams.num_cluster[index]): accp}
                metrics.update(proto_metrics)
            # average loss across all sets of prototypes
            loss_proto /= len(self.hparams.num_cluster) # Nawid -average loss across all the m different k nearest neighbours
            loss += loss_proto # Nawid - increase the loss

        additional_metrics = {'Loss':loss, 'ProtoLoss':loss_proto}
        metrics.update(additional_metrics)
        return metrics
        
    

    def on_train_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data

    def on_validation_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data
    
    def aux_data(self,dataloader):
        cluster_result = self.cluster_data(dataloader)
        return cluster_result