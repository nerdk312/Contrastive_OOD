import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import math
from tqdm import tqdm
import faiss
import numpy as np

from Contrastive_uncertainty.npid_pcl.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.npid_pcl.models.simple_memory import SimpleMemory
from Contrastive_uncertainty.npid_pcl.models.offline_label_bank import OfflineLabelMemory
from Contrastive_uncertainty.npid_pcl.utils.pl_metrics import precision_at_k
from Contrastive_uncertainty.npid_pcl.callbacks.general_callbacks import quickloading


# Based on code from https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/npid_pcl.py
class NPIDPCLModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        num_negatives: int = 8192,
        softmax_temperature: float = 0.07,
        memory_momentum = 0.5,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        use_mlp: bool = False,
        num_channels:int = 3, # number of channels for the specific dataset
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        num_cluster: list = [10],
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodule = datamodule

        self.encoder= self.init_encoders()
        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
            
        if self.hparams.pretrained_network is not None:
            self.encoder_loading(self.hparams.pretrained_network)
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder = custom_resnet18(latent_size = self.hparams.emb_dim, num_channels=self.hparams.num_channels, num_classes=10)
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder = custom_resnet50(latent_size = self.hparams.emb_dim, num_channels=self.hparams.num_channels, num_classes=10)
        
        return encoder
       
    def callback_vector(self,x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder(x)
        z = nn.functional.normalize(z, dim=1)
        return z
    
    def datasize(self, dataloader): # obtains a dataset size for the k-means based on the batch size
        batch_size = self.datamodule.batch_size
        dataset_size = len(dataloader.dataset)

        batch_num = math.floor(dataset_size/batch_size)
        new_dataset_size = batch_num * batch_size

        return int(new_dataset_size)

    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')

        
        features = torch.zeros(self.data_length, self.hparams.emb_dim, device=self.device)
        loader = quickloading(self.quick_load,dataloader) # Used to get a single batch or used to get the entire dataset
        for i, (images, labels, indices) in enumerate(tqdm(loader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder(images)  # Nawid - obtain features for the task
            feat = nn.functional.normalize(feat, dim=1) # Obtain 12 normalised features for clustering
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()
    
    def run_kmeans(self, x):
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
            cfg.useFloat16 = True  # False
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
            cluster_result['im2cluster'].append(torch.zeros(self.datasize(dataloader),dtype=torch.long,device = self.device))
            cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.hparams.emb_dim,device = self.device))
            cluster_result['density'].append(torch.zeros(int(num_cluster),device = self.device))
        
         #if using a single gpuif args.gpu == 0:
        #import ipdb; ipdb.set_trace() 
        features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
        features = features.numpy()
        # Nawid - compute K-means
        cluster_result = self.run_kmeans(features)  #run kmeans clustering on master node
        # Obtain inputs for the memory bank in the correct format
        
        cluster_labels = cluster_result['im2cluster'][0]
        indices = torch.arange(len(features), dtype=torch.long, device=self.device) # indices for the features
        # Update memory bank with initial values or moving average
        #import ipdb; ipdb.set_trace()
        if self.current_epoch ==0:
            self.memory_bank.init_memory(self,feature=features,label=cluster_labels.cpu().numpy())
        #else:
        #    self.memory_bank.update_samples_memory(indices,feature=features, label=cluster_labels)
        
        self.weights = self.set_reweight(labels=cluster_result['im2cluster'][0].cpu())
        return cluster_result

    def set_reweight(self, labels, reweight_pow=0.5):
        """Loss re-weighting.
        Re-weighting the loss according to the number of samples in each class.
        Args:
            labels (numpy.ndarray): Label assignments.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        
        hist = np.bincount(
            labels, minlength=self.hparams.num_cluster[0]).astype(np.float32)
        inv_hist = (1. / (hist + 1e-10))**reweight_pow
        weight = inv_hist / inv_hist.sum()
        # Obtain weights in tensor form
        weight = torch.from_numpy(weight).to(self.device)
        return weight

    def forward(self, img, idx,cluster_result):
        """ Forward computation

        Args:
        img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
        
        Returns:
        logits (Tensor): logits of the data

        """
        #import ipdb; ipdb.set_trace()
        features = self.encoder(img) 
        features = nn.functional.normalize(features) # BxD
        bs, feat_dim = features.shape[:2]
        # number of negatives is equal to the batch size multipled by the number of negatives for each sample
        '''
        neg_idx = self.memory_bank.multinomial.draw(bs * self.hparams.num_negatives)
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.hparams.num_negatives,
                                                    feat_dim)  # BxKxC
        '''
        '''
        neg_idx = self.memory_bank.multinomial.draw(self.hparams.num_negatives)
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.hparams.num_negatives/bs,
                                                    feat_dim)  # BxKxC
        '''        
        # Obtain positive features 
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC

        # Obtain negative features
        neg_idx = self.memory_bank.multinomial.draw(self.hparams.num_negatives)
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(self.hparams.num_negatives,
                                                    feat_dim)  # KxC

        # Obtain positive and negative logits of the data
        # shape (Bx1)
        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, features]).unsqueeze(-1)
        # shape (BxKxc and BxCx1) to give BxKx1 which is then squeezed to (BxK)
        #neg_logits = torch.bmm(neg_feat, features.unsqueeze(2)).squeeze(2)
        #import ipdb; ipdb.set_trace()
        neg_logits = torch.einsum('nc,kc->nk', [features, neg_feat.detach()]) # Nawid - negative logits (dot product between key and negative samples in a query bank)

        # Concatenate (nx1) and (nk) to get (n x k+1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        # apply temperature
        logits /= self.hparams.softmax_temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long) # Nawid - class zero is always the correct class, which corresponds to the postiive examples in the logit tensor (due to the positives being concatenated first)
        labels = labels.type_as(logits)


        # Group based contrastive learning

        # Concatenate all the examples from memory
        cluster_labels = cluster_result['im2cluster'][0]
        anchor_pseudo_labels = cluster_labels[idx]
        
        # Concatenate the instance positives (memory of anchor) as well as the negative instances to get all the different memory samples
        # (Bxc) and (Kxc) concatenate to make (B+Kxc)
        memory_feat = torch.cat((pos_feat,neg_feat),dim=0)
        # Obtain the labels of the negative memory features from the label bank of the memory bank using the negative indice
        memory_labels = torch.index_select(self.memory_bank.label_bank, 0,
                                      neg_idx)
        # Concatenate anchor labels (which are the labels of the instance positives) with the labels of the negatives
        memory_labels = torch.cat((anchor_pseudo_labels.clone(),memory_labels),dim=0) 
        loss_proto = self.supervised_contrastive_forward(features,memory_feat,anchor_pseudo_labels,memory_labels)

        # update memory bank
        with torch.no_grad():
            self.memory_bank.update_samples_memory(idx, features.detach(),anchor_pseudo_labels.detach())

        return logits, labels

    
    def supervised_contrastive_forward(self, anchor_features,memory_features, anchor_labels, memory_labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            anchor_labels: clustering labels of shape [bsz].
            memory_labels: clustering labels of shape [K].
        Returns:
            A loss scalar.
        """
        #import ipdb; ipdb.set_trace()
        batch_size = anchor_features.shape[0]
        if anchor_labels is not None:
            # Change the shape of the labels to make the values compatiable with one another
            anchor_labels = anchor_labels.contiguous().view(-1, 1)
            memory_labels = memory_labels.contiguous().view(-1,1)
            if anchor_labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # Makes a mask with values of 0 and 1 depending on whether the labels between two different samples in the batch are the same (shape [B,K])
            mask = torch.eq(anchor_labels, memory_labels.T).float().to(self.device)
        
        # compute logits (between each data point with every other data point )
        anchor_dot_contrast = torch.div(  # Nawid - similarity between the anchor and the memory features  (shape [bsz,D] x [D,K) gives [bsz,K])
            torch.matmul(anchor_features, memory_features.T),
            self.hparams.softmax_temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
    
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #import ipdb; ipdb.set_trace()
        loss = - (self.hparams.softmax_temperature / 0.07) * mean_log_prob_pos
        
        # Used to reweigh the loss by how common the different data points are
        import ipdb; ipdb.set_trace()
        weightings = torch.index_select(self.weights,0,anchor_labels.squeeze(1))
        loss = loss*weightings
        # Nawid - changes to shape (anchor_count, batch)
        loss = loss.view(1, batch_size).mean()
        return loss

    def loss_function(self, batch):
        (img_1, img_2), labels,indices = batch
        output, target = self(img=img_1,idx=indices,cluster_result=self.auxillary_data)      
        loss = F.cross_entropy(output, target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))
        metrics = {'Loss': loss, 'Instance Accuracy @ 1': acc1, 'Instance Accuracy @ 5': acc5}

        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)


    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer
    

    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

    def on_train_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
        #return self.auxillary_data

    def on_fit_start(self):
        #import ipdb; ipdb.set_trace()
        # Decides whether to test quickly or slowly
        if self.trainer.fast_dev_run:
            self.data_length = self.datamodule.batch_size
            self.quick_load = True
            self.datamodule.train_shuffle = False
        else:
            self.data_length = self.datamodule.total_dataloader_samples
            self.quick_load = False
            self.datamodule.train_shuffle = True
        
        self.memory_bank = OfflineLabelMemory(length=self.data_length, feat_dim=self.hparams.emb_dim, memory_momentum=self.hparams.memory_momentum, num_classes=10)
        # create the encoders
        # num_classes is the output fc dimension

        dataloader = self.datamodule.val_dataloader()
        self.auxillary_data = self.aux_data(dataloader)
    
    def aux_data(self,dataloader):
        cluster_result = self.cluster_data(dataloader)
        return cluster_result