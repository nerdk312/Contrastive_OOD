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
from Contrastive_uncertainty.NNCL.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50

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
        self._momentum_update_key_encoder()
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
                if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        #import ipdb; ipdb.set_trace()
        for k,v in metrics.items():
                if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
                if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)

    def loss_function(self, batch):
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

    def _momentum_update_key_encoder(self):
        raise NotImplementedError



class NNCLModule(base_module):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes:int = 10,
        class_dict:dict = None,
        emb_dim: int = 128,
        num_negatives: int = 65536, # 4096
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        num_cluster: list = [100],
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
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
        
        
        
        # Quick test of obtaining clusters (when using fast dev run)    
        #self.auxillary_data = self.aux_data()
        
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
    def compute_features(self, data): # features for clustering
        features = self.encoder_k.group_forward(data) # vector for group clustering
        features = nn.functional.normalize(features, dim=1)
        features = features.cpu() # numpy required for clustering
        return features
    
    
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
    
    def callback_vector(self,x): # Vector method which should be used in the callbacks
        z = self.encoder_k.group_forward(x)
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
            clus.min_points_per_centroid = 5

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

    def cluster_data(self,x):
        features = self.compute_features(x) # Obtain features for a single batch of data
        # placeholder for clustering result
        cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
        for num_cluster in self.hparams.num_cluster: # Nawid -Makes separate list for each different k value of the cluster (clustering is performed several times with different values of k), array of zeros for the im2cluster, the centroids and the density/concentration
            cluster_result['im2cluster'].append(torch.zeros(features.shape[0],dtype=torch.long,device = self.device))
            cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.hparams.emb_dim,device = self.device))
            cluster_result['density'].append(torch.zeros(int(num_cluster),device = self.device))
        
         #if using a single gpuif args.gpu == 0:
        features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice 
        features = features.numpy()
        # Nawid - compute K-means
        cluster_result = self.run_kmeans(features)  #run kmeans clustering on master node
        return cluster_result

    def instance_discrimination(self,im_q,im_k):
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
        return logits, labels
    '''
    def group_discrimination(self,img, cluster_result):
        proto_labels = []
        proto_logits = []
        loss_proto = 0

        features = self.encoder_q.group_forward(img)
        features = nn.functional.normalize(features, dim=1)


        for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
            # shape of prototypes is (cluster_num,embedding size)
            # (im2cluster :shape [batch], numbers from 0 to clunster num)
            similarity = torch.mm(features, prototypes.t()) # Measure similarity between features from online encoder and prototypes from the other encoder
            # similarity shape (batch size, cluster num)
            
            proto_loss = nn.CrossEntropyLoss()(similarity,im2cluster)

            loss_proto += proto_loss
        
        loss_proto = loss_proto/(len(self.hparams.num_cluster))
        return loss_proto

    def loss_function(self,batch):
        #metrics = {}
        (img_1,img_2), labels,indices = batch
        # obtains cluster centroids information for the first view and the second view
        cluster_view_1,cluster_view_2 = self.cluster_data(img_1), self.cluster_data(img_2)

        output, target = self.instance_discrimination(im_q=img_1,im_k=img_2)

        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss

        # ProtoNCE loss
        loss_proto_view_1 = self.group_discrimination(img_2,cluster_view_1)
        loss_proto_view_2 = self.group_discrimination(img_1,cluster_view_2)
        loss_proto = (loss_proto_view_1 + loss_proto_view_2)/2

        loss = loss_instance  + loss_proto
        
        metrics = {'Loss' : loss, 'Instance Loss' : loss_instance, 'Proto Loss' : loss_proto}
        return metrics
    '''
    
    def group_discrimination(self,img, cluster_result):
        proto_labels = []
        proto_logits = []

        features = self.encoder_q.group_forward(img)
        features = nn.functional.normalize(features, dim=1)


        for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
            # shape of prototypes is (cluster_num,embedding size)
            # (im2cluster :shape [batch], numbers from 0 to clunster num)
            similarity = torch.mm(features, prototypes.t()) # Measure similarity between features from online encoder and prototypes from the other encoder
            # similarity shape (batch size, cluster num)
            
            proto_labels.append(im2cluster)
            proto_logits.append(similarity)
        
        return proto_logits, proto_labels

    def loss_function(self,batch):
        metrics = {}
        (img_1,img_2), labels,indices = batch
        # obtains cluster centroids information for the first view and the second view
        cluster_view_1,cluster_view_2 = self.cluster_data(img_1), self.cluster_data(img_2)

        output, target = self.instance_discrimination(im_q=img_1,im_k=img_2)
        proto_logits_1, proto_labels_1 = self.group_discrimination(img_2, cluster_view_1)
        proto_logits_2, proto_labels_2 = self.group_discrimination(img_1, cluster_view_2)

        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss 
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        instance_metrics = {'Instance Accuracy @ 1': acc1, 'Instance Accuracy @ 5':acc5}
        metrics.update(instance_metrics)

        loss_proto = 0
        loss_proto_view_1 = 0
        loss_proto_view_2 = 0
        for index, (proto_out_1, proto_target_1,proto_out_2,proto_target_2) in enumerate(zip(proto_logits_1,proto_labels_1,proto_logits_2,proto_labels_2)):
            # Calculate loss and accuracy for view 1
            loss_proto_view_1 += F.cross_entropy(proto_out_1,proto_target_1)
            accp_view_1 = precision_at_k(proto_out_1, proto_target_1)[0]
            # Calculate loss and accuracy for view 2
            loss_proto_view_2 += F.cross_entropy(proto_out_2,proto_target_2)
            accp_view_2 = precision_at_k(proto_out_2, proto_target_2)[0]
            # Calculate avg accuracy across views
            avg_accp = (accp_view_1 + accp_view_2)/2

            proto_metrics = {'Accuracy @ 1 '+str(self.hparams.num_cluster[index]): avg_accp}
            metrics.update(proto_metrics)

        # Find average of each view across the different clusterings
        loss_proto_view_1 /= len(self.hparams.num_cluster)
        loss_proto_view_2 /= len(self.hparams.num_cluster)

        loss_proto = (loss_proto_view_1 + loss_proto_view_2)/2
        loss = loss_instance + loss_proto

        loss_metrics = {'Loss' : loss, 'Instance Loss' : loss_instance, 'Proto Loss' : loss_proto}
        metrics.update(loss_metrics)
        
        return metrics
