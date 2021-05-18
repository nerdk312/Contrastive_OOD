import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from tqdm import tqdm
import faiss

from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean



class HPCLCentroidToy(Toy):
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
        contrast_mode : str = 'one',
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

    
    @property
    def name(self):
        ''' return name of model'''
        return 'HPCL'

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

    

    def instance_forward(self, q, k):
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

    def proto_forward(self,features,labels,prototypes):
        proto_labels = []
        proto_logits = []

        logits_proto = torch.mm(features,prototypes.t().to(self.device)) # [bxd] by [dxprotosize] = [bx protosize]
        logits_proto /= 0.07  # Divide by temperature term
        proto_labels.append(labels.to(self.device))
        proto_logits.append(logits_proto)
        #import ipdb; ipdb.set_trace()
        return proto_logits, proto_labels



    
    @torch.no_grad()
    def compute_features(self, dataloader):
        print('Computing features ...')
        
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device=self.device)
        # Collate the coarse and fine grained labels
        collated_fine_labels =  torch.zeros(len(dataloader.dataset), device = self.device,dtype=torch.long)
        collated_coarse_labels =  torch.zeros(len(dataloader.dataset), device = self.device,dtype=torch.long)
        
        for i, (images, coarse_labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            fine_labels = torch.randint(low=0, high=4, size = coarse_labels.shape)
            images, coarse_labels, fine_labels = images.to(self.device), coarse_labels.to(self.device), fine_labels.to(self.device)
                

            collated_coarse_labels[indices] = coarse_labels
            collated_fine_labels[indices] = fine_labels

            feat = self.encoder_k(images)   # Nawid - obtain momentum features
            features[indices] = feat  # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        # Collate the coarse and the fine grain labels   
        collated_labels = [collated_fine_labels, collated_coarse_labels]
        features = nn.functional.normalize(features, dim=1)
        return features, collated_labels
    
    
    def feature_vector(self, data):
        z = self.encoder_k(data)
        return z

    def loss_function(self,batch):
        metrics = {}
        # output images as well as coarse labels of the data
        (img_1,img_2), coarse_labels, indices = batch
        
        # Made random fine labels for testing purposes 
        fine_labels = torch.randint(low=0, high=4, size = coarse_labels.shape)
        collated_labels = [fine_labels,coarse_labels]
        #import ipdb; ipdb.set_trace()
        q = self.encoder_q(img_1)
        q = nn.functional.normalize(q, dim=1)
        k = self.encoder_q(img_2)
        k = nn.functional.normalize(k, dim=1)
        output, target = self.instance_forward(q,k)
        # InfoNCE loss
        loss_instance = F.cross_entropy(output, target) # Nawid - instance based info NCE loss
        acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        instance_metrics = {'Instance Loss': loss_instance, 'Instance Accuracy @1':acc_1,'Instance Accuracy @5':acc_5}
        metrics.update(instance_metrics)

        
        # Initialise loss value
        loss_proto = 0 


        
        proto_out, proto_target = self.proto_forward(q,coarse_labels,self.auxillary_data)
        loss_proto += F.cross_entropy(proto_out, proto_target)
        accp = precision_at_k(proto_out, proto_target)[0]
        proto_metrics = {'Proto Loss': loss_proto,' Proto Accuracy @ 1' : accp}
        metrics.update(proto_metrics)

        '''
        for index, data_labels in enumerate(collated_labels):
            loss_proto += self.supervised_contrastive_forward(features=features,labels=data_labels)
        
        # Normalise the proto loss by number of different labels present
        loss_proto /= len(collated_labels)
        '''
        '''
            for index, (proto_out,proto_target) in enumerate(zip(output_proto, target_proto)): # Nawid - I believe this goes through the results of the m different k clustering results
                loss_proto += F.cross_entropy(proto_out, proto_target) #
                accp = precision_at_k(proto_out, proto_target)[0]
               # acc_proto.update(accp[0], images[0].size(0))
                # Log accuracy for the specific case
                proto_metrics = {'Accuracy @ 1 Cluster '+str(self.hparams.num_cluster[index]): accp}
                metrics.update(proto_metrics)
            # average loss across all sets of prototypes
            loss_proto /= len(self.hparams.num_cluster) # Nawid -average loss across all the m different k nearest neighbours
        '''     
        loss = loss_instance + loss_proto # Nawid - increase the loss

        additional_metrics = {'Loss':loss}
        metrics.update(additional_metrics)
        return metrics
    
    @torch.no_grad()
    def obtain_protoypes(self,features,labels):
        #y = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        # Labels is a tensor of the different values
        collated_prototypes = []
        for i, label in enumerate(labels):
            y = F.one_hot(labels.long()).float()
            # compute sum of embeddings on class by class basis
            #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
            #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
            features_sum = torch.matmul(y.T,features) # (num_classes,batch) (batch,features) to get (num_class, features)

            #features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
            prototypes = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes
            prototypes = prototypes.T # Turn back into shape (num_classes,features)
            # Collate the prototypes for the coarse and fine grain labels case
            collated_prototypes.append(prototypes)
        
        return collated_prototypes
    
    '''
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
    '''
    
    def aux_data(self,dataloader):
        features, collated_labels = self.compute_features(dataloader)
        prototypes = self.obtain_protoypes(features, collated_labels)
        return prototypes

    def on_train_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        #import ipdb; ipdb.set_trace()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data

    def on_fit_start(self):
        if self.trainer.testing:
            dataloader = self.datamodule.test_dataloader()
        else:
            dataloader = self.datamodule.val_dataloader()
        
        self.auxillary_data = self.aux_data(dataloader)
