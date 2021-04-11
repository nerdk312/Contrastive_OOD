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
        

        self.auxillary_data = None #self.aux_data() #self.on_train_epoch_start(self.datamodule)
        
        
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




class PCLModule(base_module):
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
                pos_proto_id = im2cluster[index] # Nawid - get the true cluster assignment for each of the different samples
                pos_prototypes = prototypes[pos_proto_id] # Nawid- prototypes is a kxd array of k , d dimensional clusters. Therefore this chooses the true clusters for the positive samples. Therefore this is a [B x d] matrix

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
        #import ipdb; ipdb.set_trace()
        print('Computing features ...')
        #import ipdb;ipdb.set_trace()
        
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device = self.device)

        #import ipdb; ipdb.set_trace()
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            assert len(dataloader) >0, "dataloader is empty"
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
                #import ipdb; ipdb.set_trace()
                images = images.to(self.device)
            
            feat = self(images,is_eval=True)   # Nawid - obtain momentum features
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()
    
    def feature_vector(self,x):
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder_q.representation_output(x)
        z = nn.functional.normalize(z, dim=1)
        return z
    
    def feature_vector_compressed(self,x):
        """
        Input:
            x: a batch of images for classification
        Output:
            logits
        """
        # compute query features
        z = self.feature_vector(x) # Gets the feature map representations which I use for the purpose of pretraining
        z = self.encoder_q.class_fc1(z)
        z = nn.functional.normalize(z, dim=1)
        return z
    
    def class_discrimination(self, x):
        """
        Input:
            x: a batch of images for classification
        Output:
            logits
        """
        # compute query features
        z = self.feature_vector(x) # Gets the feature map representations which I use for the purpose of pretraining
        z = F.relu(self.encoder_q.class_fc1(z))

        if self.hparams.normalize:
            z = nn.functional.normalize(z, dim=1)

        logits = self.encoder_q.class_fc2(z)
        return logits


    def classification_loss(self, batch,auxillary_data = None):
        (img_1, img_2), labels,indices = batch
        logits = self.class_discrimination(img_1)
        loss = F.cross_entropy(logits.float(), labels.long())
        class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
        metrics = {'Class Loss': loss, 'Class Accuracy @ 1': class_acc1, 'Class Accuracy @ 5': class_acc5}

        return metrics

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
    
    # Override the other loss to use the test step as well as the batch size for the specific task
    def test_step(self, batch, batch_idx):
        loss = torch.tensor([0.0], device=self.device)

        metrics = self.loss_function(batch,self.auxillary_data)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        loss += metrics['Loss']
        
        metrics = self.classification_loss(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        loss += metrics['Class Loss']
        self.log('Test Total Loss', loss.item(),on_epoch=True)
        
        #return {'logits':logits,'target':labels} # returns y_pred as y_pred are essentially the logits in this case, and I want to log how the logits change in time
    '''
    def test_epoch_end(self, test_step_outputs):
        # Saves the predictions which are the logits in this case to see how the logits are changing in time
        
        #flattened_logits = torch.flatten(torch.cat([output['logits']for output in validation_step_outputs])) #  concatenate the logits
        #self.logger.experiment.log(
        #    {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
        #    "global_step": self.global_step})
        
        preds = torch.cat([output['logits'] for output in test_step_outputs])
        targets = torch.cat([output['target'] for output in test_step_outputs])

        top_pred_ids = preds.argmax(axis=1)

        self.logger.experiment.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(probs = preds.cpu().numpy(),
            preds=None, y_true=targets.cpu().numpy(),
            class_names=self.class_names),
            "global_step": self.global_step
                  })
    '''    
    def aux_data(self):        
        dataloader = self.datamodule.val_dataloader()
        cluster_result = self.cluster_data(dataloader)
        return cluster_result

    # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_validation_start 
    #Do not currently need to obtain the auxillary data of the train dataloader as the val and train data are the same in PCL currently
    '''
    def on_epoch_start(self):
        self.auxillary_data = self.aux_data()
        return self.auxillary_data
    '''    
    
    def on_train_epoch_start(self):
        self.auxillary_data = self.aux_data()
        return self.auxillary_data

    def on_validation_epoch_start(self):
        self.auxillary_data = self.aux_data()
        return self.auxillary_data
            
    # Loads both network as a target state dict
    def encoder_loading(self,pretrained_network):
        print('checkpoint loaded')
        checkpoint = torch.load(pretrained_network)
        self.encoder_q.load_state_dict(checkpoint['target_encoder_state_dict'])
        self.encoder_k.load_state_dict(checkpoint['target_encoder_state_dict'])


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output