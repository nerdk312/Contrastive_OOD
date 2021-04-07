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



class PCLToy(Toy):
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
        num_cluster :list = [100],
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

        self.auxillary_data = self.aux_data()

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
                
                # get positive prototypes
                pos_proto_id = im2cluster[index] # Nawid - get the true cluster assignment for each of the different samples
                pos_prototypes = prototypes[pos_proto_id] # Nawid- prototypes is a kxd array of k , d dimensional clusters. Therefore this chooses the true clusters for the positive samples. Therefore this is a [B x d] matrix

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())] # Nawid - obtains all the cluster ids which were present
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist()) # Nawid - all the negative clusters are the set of all prototypes minus the set of all the negative prototypes
                neg_proto_id = sample(neg_proto_id, self.hparams.num_negatives) #sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id] # Nawid - sample negative prototypes

                proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0) # Nawid - concatenate positive and negative prototypes, so this is  a [bxd] concatenated with [rxd] to make a [b + r xd]

                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t().to(self.device)) # Nawid - dot product between query and the prototypes (where the selected prototypes are transposed). The matrix multiplication is  [b x d] . [dx b +r] to make a [b x b +r]

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0),device = self.device).long()# Nawid - targets for the prototypes, this is a 1D vector with values from 0 to q-1 which represents that the value which shows that the diagonal should be the largest value

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id)],dim=0)].to(self.device)
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
        import ipdb; ipdb.set_trace()
        for i, (images,labels, indices) in enumerate(tqdm(dataloader)):
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
        (img_1,img_2), labels,indices = batch

        # compute output -  Nawid - obtain instance features and targets as  well as the information for the case of the proto loss
        output, target, output_proto, target_proto = self(im_q=img_1, im_k=img_2, cluster_result=cluster_result, index=indices) # Nawid- obtain output

        # InfoNCE loss
        loss = F.cross_entropy(output, target) # Nawid - instance based info NCE loss

        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out,proto_target in zip(output_proto, target_proto): # Nawid - I believe this goes through the results of the m different k clustering results
                loss_proto += F.cross_entropy(proto_out, proto_target) #
                accp = precision_at_k(proto_out, proto_target)[0]
               # acc_proto.update(accp[0], images[0].size(0))

            # average loss across all sets of prototypes
            loss_proto /= len(self.hparams.num_cluster) # Nawid -average loss across all the m different k nearest neighbours
            loss += loss_proto # Nawid - increase the loss

        return loss

    def aux_data(self):
        dataloader = self.datamodule.val_dataloader()
        cluster_result = self.cluster_data(dataloader)
        return cluster_result

    def on_train_epoch_end(self, outputs):
        self.auxillary_data = self.aux_data()
        return self.auxillary_data
    '''
    def train_epoch_start(self, epoch):

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # structured result accumulators for callbacks
        self.early_stopping_accumulator = Accumulator()
        self.checkpoint_accumulator = Accumulator()

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

        self.auxillary_data = self.aux_data()
    '''
    '''
    def on_train_epoch_start(self):
        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

        #  (The only part that I added) - Used to call the cluster results for the case of the prototypical 
        self.auxillary_data = self.aux_data()
        
    '''
    







