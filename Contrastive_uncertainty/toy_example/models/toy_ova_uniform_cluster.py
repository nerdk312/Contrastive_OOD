import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import faiss
import numpy as np

from Contrastive_uncertainty.toy_example.models.toy_encoder import Backbone
from Contrastive_uncertainty.Moco.pl_metrics import precision_at_k
from Contrastive_uncertainty.toy_example.models.toy_module import Toy

class OVAUniformClusterToy(Toy):
    def __init__(self,
        datamodule,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        hidden_dim: int =  20,
        emb_dim: int = 2,
        num_classes:int = 2,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        num_cluster :list = [10],
        ):
        super().__init__(datamodule, optimizer, learning_rate,
                         momentum, weight_decay)
        self.save_hyperparameters()
        
        
        # Nawid - required to use for the fine tuning

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders()
        self.classifier = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    # Instantiate classifier
    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder_q = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        encoder_k = Backbone(self.hparams.hidden_dim, self.hparams.emb_dim)
        return encoder_q, encoder_k
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
    
    def loss_function(self, batch, auxillary_data):
        (img_1, img_2), labels, indices = batch
        one_hot_labels = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        centroids = self.update_embeddings(img_1, labels)
        y_pred = self(img_1, centroids)
        distance_loss = F.binary_cross_entropy(y_pred, one_hot_labels)
        acc1, = precision_at_k(y_pred, labels)

        z =  self.encoder_q(img_1)
        uniformity_loss = self.uniform_loss(z)

        #cluster_result = self.cluster_data(img_1)
        #import ipdb; ipdb.set_trace()
        proto_logits, proto_labels = self.group_discrimination(img_1, cluster_result=auxillary_data,index = indices)
        loss_proto = 0
        
        for index, (proto_out, proto_target) in enumerate(zip(proto_logits, proto_labels)):
            loss_proto += F.cross_entropy(proto_out, proto_target)

        # Find average of each view across the different clusterings
        loss_proto /= len(self.hparams.num_cluster)
        
        loss =  loss_proto  #0.8*distance_loss + 0.1*uniformity_loss + 1.0*loss_proto

        #loss = 0.8*distance_loss + 0.2*uniformity_loss
        
        '''
        correct = torch.argmax(y_pred[:original_xs_length].detach(),dim=1).view(original_xs_length,-1) == labels # look at calculating the correct values only for the case of the true data
        accuracy = torch.mean(correct.float())
        '''
        metrics = {'Loss': loss, 'Distance Loss': distance_loss,'Uniformity Loss':uniformity_loss,'Accuracy @ 1': acc1}
        return metrics

    def feature_vector(self, x): # Obtain feature vector
        x = self.encoder_k(x)
        #x = nn.functional.normalize(x, dim=1)
        return x
    
    # Uniformity and alignment
    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, x, centroids):  # obtain predictions
        z = self.encoder_q(x)
        distances = self.euclidean_dist(z, centroids)
        
        y_pred = 2*torch.sigmoid(distances)
        return y_pred  # shape (batch,num_classes)

    def class_discrimination(self, x, centroids): # same as forward
        y_pred = self(x, centroids)
        return y_pred
    
    def centroid_confidence(self, x, centroids): # same as forward
        y_pred = self(x, centroids)
        return y_pred
    
    def euclidean_dist(self, x, y):  # Calculates the difference
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)  # shape (batch,num class, features)
        y = y.unsqueeze(0).expand(n, m, d)  # shape (batch,num class, features)
        diff = x - y
        distances = -torch.pow(diff, 2).sum(2)  # Need to get the negative distance , SHAPE (batch, num class)
        return distances
    
    @torch.no_grad()
    def update_embeddings(self, x, labels): # Assume y is one hot encoder
        z = self.encoder_k(x)  # (batch,features) # use momentum encoder to get features
        y = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
        # compute sum of embeddings on class by class basis

        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        features_sum = torch.matmul(y.T,z) # (num_classes,batch) (batch,features) to get (num_class, features)

        #features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
        

        embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings
    
        
    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device = self.device)
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder_k(images)  # Nawid - obtain features for the task
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()
    
     
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

    
    
    # Obtain the logits for a particular class for the network
    def group_discrimination(self,img, cluster_result,index):
        proto_labels = []
        proto_logits = []

        features = self.encoder_q(img)
        features = nn.functional.normalize(features, dim=1)

        for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])): # Nawid - go through a loop of the results of the k-nearest neighbours (m different times)
            # shape of prototypes is (cluster_num,embedding size)
            # (im2cluster :shape [batch], numbers from 0 to clunster num)
            #prototypes = prototypes[index] # Obtain the class prototypes only for the specific data points
            similarity = torch.mm(features, prototypes.t()) # Measure similarity between features from online encoder and prototypes from the other encoder
            # similarity shape (batch size, cluster num)
            similarity /= self.hparams.softmax_temperature

            proto_labels.append(im2cluster[index]) # Obtain the labels for the specific indices of the dataset
            proto_logits.append(similarity)
        
        return proto_logits, proto_labels
    
    def on_validation_epoch_start(self):
        dataloader = self.datamodule.val_dataloader()
        #import ipdb; ipdb.set_trace()
        self.auxillary_data = self.aux_data(dataloader)
        return self.auxillary_data
    
    def aux_data(self,dataloader):
        cluster_result = self.cluster_data(dataloader)
        return cluster_result
     
   
    '''
    @torch.no_grad()
    def compute_features(self,dataloader):
        print('Computing features ...')
        features = torch.zeros(len(dataloader.dataset), self.hparams.emb_dim, device = self.device)
        for i, (images, labels, indices) in enumerate(tqdm(dataloader)):
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_images = images
            images = images.to(self.device)

            feat = self.encoder_k(images)  # Nawid - obtain features for the task
            features[indices] = feat # Nawid - place features in matrix, where the features are placed based on the index value which shows the index in the training data
        return features.cpu()
    '''


    '''

    def cluster_data(self,dataloader):
        features = self.compute_features(dataloader)
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
    


     @torch.no_grad()
    def compute_features(self, data): # features for clustering
        features = self.encoder_k(data) # vector for group clustering
        features = nn.functional.normalize(features, dim=1)
        features = features.cpu() # numpy required for clustering
        return features

    '''