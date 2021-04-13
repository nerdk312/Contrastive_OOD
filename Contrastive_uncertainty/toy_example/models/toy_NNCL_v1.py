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

        
        self.encoder_q, self.encoder_k = self.init_encoders()

        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.pretrained_network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(self.hparams.hidden_dim,self.hparams.emb_dim)
        encoder_k = Backbone(self.hparams.hidden_dim,self.hparams.emb_dim)
        
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

    
    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
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

        if is_eval: # Nawid - obtain key outputs
            k = self.encoder_k(im_q)
            k = nn.functional.normalize(k, dim=1) # Nawid - return normalised momentum embedding
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder


            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1) # Nawid - normalised key embeddings

            

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1) # Nawid - normalised query embeddings

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #Nawid - positive logit between output of key and query
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # Nawid - negative logits (dot product between key and negative samples in a query bank)

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1) # Nawid - total logits - instance based loss to keep property of local smoothness

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long,device = self.device)
        #labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k) # Nawid - queue values
         



        # prototypical contrast - Nawid - performs the protoNCE
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
                #import ipdb; ipdb.set_trace()
                # get positive prototypes
                pos_proto_id = im2cluster[index] # Nawid - get the assignment for which cluster the image is present (corresponding to index)
                import ipdb; ipdb.set_trace()
                pos_prototypes = prototypes[pos_proto_id] # Nawid- prototypes shape [batch,embedding] is a kxd array of k , d dimensional clusters. Therefore this chooses the true clusters for the positive samples. Therefore this is a [B x d] matrix

                # Need to change pos prototypes to take into account different samples, also need to associate the values in the queue with a cluster index also

                # 
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())] # Nawid - obtains all the cluster ids which were present
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist()) # Nawid - all the negative clusters are the set of all prototypes minus the set of all the negative prototypes
                neg_proto_id = sample(neg_proto_id, self.hparams.num_negatives) #sample r negative prototypes
                #neg_proto_id = neg_proto_id.to(self.device)
                neg_prototypes = prototypes[neg_proto_id] # Nawid - sample negative prototypes

                proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0) # Nawid - concatenate positive and negative prototypes, so this is  a [bxd] concatenated with [rxd] to make a [b + r xd]

                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t().to(self.device)) # Nawid - dot product between query and the prototypes (where the selected prototypes are transposed). The matrix multiplication is  [b x d] . [dx b +r] to make a [b x b +r]

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0),device = self.device).long()# Nawid - targets for the prototypes, this is a 1D vector with values from 0 to q-1 which represents that the value which shows that the diagonal should be the largest value

                # scaling temperatures for the selected prototypes
                #import ipdb; ipdb.set_trace()
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).to(self.device)], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None

    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')
        #import ipdb;ipdb.set_trace()
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device = self.device)
        #import ipdb; ipdb.set_trace()
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
                images = images.to(self.device)

            feat = self(images,is_eval=True)   # Nawid - obtain momentum features
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()


    def feature_vector(self, data):
        z = self.encoder_q(data)
        return z

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
        metrics = {}
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

    def aux_data(self,dataloader):
        cluster_result = self.cluster_data(dataloader)
        return cluster_result

    # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_validation_start 
    #Do not currently need to obtain the auxillary data of the train dataloader as the val and train data are the same in PCL currently
    '''
    def on_train_epoch_start(self):
        dataloader = self.datamodule.train_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data
    '''
        
    def on_validation_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        #import ipdb; ipdb.set_trace()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data
    
    
    # SUP CON
    # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
    # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
    def supcon_forward(self, features, labels=None, mask=None):
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
        loss = - (self.hparams.softmax_temperature / self.hparams.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
    
    def supcon_loss_function(self, batch ,auxillary_data=None):
        (img_1, img_2), labels, indices = batch
        imgs = torch.cat([img_1, img_2], dim=0)
        bsz = labels.shape[0]
        features = self.encoder(imgs)
        ft_1, ft_2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([ft_1.unsqueeze(1), ft_2.unsqueeze(1)], dim=1)
        loss = self.forward(features, labels)
        metrics = {'Loss':loss}

        return metrics
    
    
        
    

    # AUXILLARY FUNCTION FOR CALLBACK, not critical to function of network

    def class_discrimination(self, data):
        z = self.encoder(data)
        z = F.relu(z)
        logits = self.classifier(z)
        return logits
    
    @torch.no_grad()
    def update_embeddings(self, x, labels): # Assume y is one hot encoder
        z = self.feature_vector(x)  # (batch,features)
        y = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        # compute sum of embeddings on class by class basis

        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        features_sum = torch.matmul(y.T,z) # (num_classes,batch) (batch,features) to get (num_class, features)

        #features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
        

        embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings
    

    def euclidean_dist(self, x, y):  # Calculates the difference
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        diff = x - y
        distances = -torch.pow(diff, 2).sum(2)  # Need to get the negative distance
        return distances
    
    def centroid_confidence(self, x,centroids):
        z = self.feature_vector(x)
        distances = self.euclidean_dist(z, centroids)
        return distances  # shape: (batch, num classes)




