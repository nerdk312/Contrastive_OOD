import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import sklearn.metrics as skm

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.Moco.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.Moco.loss_functions import class_discrimination


class ReliabiltyLogger(pl.Callback):
    def __init__(self,samples,num_samples=300):
        super().__init__()
        self.imgs, self.labels, self.indices = samples
        if isinstance(self.imgs,tuple) or isinstance(self.imgs,list):
            self.imgs, *aug_imgs = self.imgs

        self.imgs = self.imgs[:num_samples]
        self.labels = self.labels[:num_samples]

    def on_test_epoch_end(self, trainer, pl_module):
        #print('Reliability logging training?:',pl_module.encoder_q.training)
        imgs = self.imgs.to(device=pl_module.device)
        labels = self.labels.to(device=pl_module.device)
        # Make target centroids
        #one_hot_labels = F.one_hot(labels).float()

        logits = class_discrimination(pl_module,imgs)
        y_pred = F.softmax(logits, dim=1)
        ece = self.make_model_diagrams(y_pred, labels, pl_module)# calculates ECE as well as makes reliability diagram


    def calculate_ece(self, logits, labels, pl_module, n_bins=10):
        """
        Calculates the Expected Calibration Error of a model.
        (This isn't necessary for temperature scaling, just a cool metric).
        The input to this loss is the logits of a model, NOT the softmax scores.
        This divides the confidence outputs into equally-sized interval bins.
        In each bin, we compute the confidence gap:
        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
        We then return a weighted average of the gaps, based on the number
        of samples in each bin
        See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
        2015.
        """


        bin_boundaries = torch.linspace(0, 1, n_bins + 1) # Makes bins
        bin_lowers = bin_boundaries[:-1] # Lower boundary of bin
        bin_uppers = bin_boundaries[1:] # Upper boundary of bin

        #softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(logits, 1)  # Nawid - obtain the confidences and the indices of confidence which are the predictions

        accuracies = predictions.eq(labels)  # Calculate accuracies

        ece = torch.zeros(1,device = pl_module.device)#.cuda()#, device=pl_module.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers): # Iterate through the bin intervals
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) # Looks  values of confidences which are greater bin lower and look at confidences which are less than bin_upper, and gets true false values. Then multiples them to perform and operation
            prop_in_bin = in_bin.float().mean() # gets all the true and false values and finds the mean to see the proportion of values in this bin
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()# Calculates the mean accuracy of all the indices which are in this bin
                avg_confidence_in_bin = confidences[in_bin].mean() # Calculates the mean confidence of all the indices which are in this bin
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin # Calculates the ece contribution of this bin
        return ece.item()

    def make_model_diagrams(self,outputs, labels,pl_module, n_bins=10):
        """
        outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
        - NOT the softmaxes
        labels - a torch tensor (size n) with the labels
        """
        # Calculate confidences and accuracies
        #softmaxes = torch.nn.functional.softmax(outputs, 1)
        confidences, predictions = outputs.max(1) # Calculate the confidences and the predictions
        #predictions = predictions.to(pl_module.device)

        accuracies = torch.eq(predictions, labels) # Calculate accuracy in true and false (0 or 1s)
        overall_accuracy = (predictions==labels).sum().item()/len(labels) # Calculate number for the accuracy from the sum of the zeros and ones

        # Reliability diagram
        # Generate bins
        bins = torch.linspace(0, 1, n_bins + 1) # generate bins
        width = 1.0 / n_bins # width of each bin
        bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2 # bin centres
        bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]  # Get the indices for the confidenes which belong to a particular bin

        # Get bin accuracies and scores
        bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])  # Get accuracies for individual bins by looking at indices which are in the bin
        bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices]) # Get scores for individual bins by looking at indices which are in the bin
        # Obtain the gap between scores and the accuracies and show on plot
        plt.figure(0, figsize=(8, 8))
        gap = (bin_scores - bin_corrects) # Gap between the accuracy of a bin and the scores(confidences)
        confs = plt.bar(bin_centers, bin_corrects, width=width, alpha=0.1, ec='black')
        gaps = plt.bar(bin_centers, (bin_scores - bin_corrects), bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

        # Calculate ece value
        ece = self.calculate_ece(outputs, labels,pl_module)

        # Clean up
        bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
        plt.text(0.2, 0.85, "ECE: {:.2f}".format(ece), ha="center", va="center", size=20, weight = 'bold', bbox=bbox_props) # Text for ECE value

        # Labels and limits
        plt.title("Reliability Diagram", size=20)
        plt.ylabel("Accuracy (P[y]",  size=18)
        plt.xlabel("Confidence",  size=18)
        plt.xlim(0,1)
        plt.ylim(0,1)


        # Save diagram into wandb

        reliability_filename = 'Images/reliability.png'
        plt.savefig(reliability_filename)
        wandb_reliability = 'reliability'
        wandb.log({wandb_reliability:wandb.Image(reliability_filename)})
        plt.close()

        wandb.run.summary["ECE"] = ece
        return ece




# Used to save the model in the directory as well as in wandb
class ModelSaving(pl.Callback):
    def __init__(self,interval):
        super().__init__()
        self.interval = interval
        self.counter = interval
        #self.epoch_last_check = 0
    # save the state dict in the local directory as well as in wandb
    '''
    def on_validation_epoch_end(self,trainer,pl_module): # save every interval
        epoch = trainer.current_epoch 
        if epoch > self.counter:
            self.save_model(pl_module,epoch)
            self.counter += self.interval # Increase the interval
    '''            
    
    def on_test_epoch_end(self, trainer, pl_module): # save during the test stage
        epoch =  trainer.current_epoch
        self.save_model(pl_module,epoch)

        '''
        if (epoch - self.epoch_last_check) < self.interval-1: #  Called at the end of training
            # Skip
            return
        
        self.save_model(pl_module,epoch)
        self.epoch_last_check = epoch
        '''
    
    def save_model(self,pl_module,epoch):
        filename = f"CurrentEpoch:{epoch}_" + wandb.run.name + '.pt' 
        print('filename:',filename)
        torch.save({
            'online_encoder_state_dict':pl_module.encoder_q.state_dict(),
            'target_encoder_state_dict':pl_module.encoder_k.state_dict(),
        },filename)
        wandb.save(filename)

    #def on_test_epoch_end(self, trainer, pl_module):
        
# Calculation of MMD based on the definition 2/ equation 1 in the paper        
# http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchetal12.pdf?origin=publication_detail
class MMD_distance(pl.Callback):
    def __init__(self,Datamodule,quick_callback):
        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback
        self.log_name = 'MMD_distance'
    '''
    def on_test_epoch_end(self,trainer, pl_module):
        self.calculate_MMD(pl_module)
    '''

    # Log MMD whilst the network is training
    def on_validation_epoch_end(self,trainer,pl_module):
        self.calculate_MMD(pl_module)

    
    def calculate_MMD(self,pl_module):
        dataloader = self.Datamodule.train_dataloader()
        with torch.no_grad():
            MMD_values = []
            low = torch.tensor(-1.0).to(device = pl_module.device)
            high = torch.tensor(1.0).to(device = pl_module.device)
            uniform_distribution = torch.distributions.uniform.Uniform(low,high) # causes all samples to be on the correct device when obtainig smaples https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
            #uniform_distribution =  torch.distributions.uniform.Uniform(-1,1).sample(output.shape)
            loader = quickloading(self.quick_callback,dataloader) # Used to get a single batch or used to get the entire dataset
            for data, target,indices in loader:
                assert len(loader)>0, 'loader is empty'
                if isinstance(data, tuple) or isinstance(data, list):
                    data, *aug_data = data # Used to take into accoutn whether the data is a tuple of the different augmentations

                data = data.to(pl_module.device)

                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = pl_module.callback_vector(data) # representation of the data
                
                uniform_samples = uniform_distribution.sample(output.shape)
                uniform_samples = torch.nn.functional.normalize(uniform_samples,dim=1) # obtain normalized representaitons on a hypersphere
                # calculate the difference between the representation and samples from a unifrom distribution on a hypersphere
                diff = output - uniform_samples
                MMD_values.append(diff.cpu().numpy())


        MMD_list = np.concatenate(MMD_values)
        MMD_dist = np.mean(MMD_list)
        # Logs the MMD distance into wandb
        wandb.log({self.log_name:MMD_dist})

        return MMD_dist

class Uniformity(pl.Callback):
    def __init__(self, t,datamodule,quick_callback):
        super().__init__()
        self.t  = t
        self.datamodule = datamodule
        self.quick_callback = quick_callback
        self.log_name = 'uniformity'
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)
    '''
    

    def on_validation_epoch_end(self,trainer,pl_module):
        features = self.obtain_features(pl_module) 
        uniformity = self.calculate_uniformity(features)

    def obtain_features(self,pl_module):
        features = []
        dataloader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            img = img.to(pl_module.device)
            features.append(pl_module.callback_vector(img))
        
        features = torch.cat(features) # Obtaint the features for the representation
        return features

    def calculate_uniformity(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features) # Change to a torch tensor

        uniformity = torch.pdist(features, p=2).pow(2).mul(-self.t).exp().mean().log()
        wandb.log({self.log_name:uniformity.item()})
        #uniformity = uniformity.item()
        return uniformity


class Centroid_distance(pl.Callback):
    def __init__(self,datamodule,quick_callback):
        super().__init__()
        self.datamodule = datamodule
        self.quick_callback = quick_callback
        self.log_distance_name = 'centroid_distance'
        self.log_rbf_similarity_name = 'centroid_rbf_similarity'
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        optimal_centroids = self.optimal_centroids(pl_module)
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        collated_distances = []
        collated_rbf_similarity = []
        for step, (data,labels) in enumerate(loader):
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
                data, labels =  data.to(pl_module.device), labels.to(pl_module.device)
                latent_vector = pl_module.feature_vector(data)

            distances = self.distance(pl_module,latent_vector,optimal_centroids)
            #import ipdb;ipdb.set_trace()
            labels = labels.reshape(len(labels),1)
            # gather takes the values from distances along dimension 1 based on the values of labels
            class_distances = torch.gather(distances,1,labels)
            class_rbf_sim = self.rbf_similarity(class_distances)

            collated_distances.append(class_distances)
            collated_rbf_similarity.append(class_rbf_sim)

        collated_distances = torch.cat(collated_distances) # combine all the values
        collated_rbf_similarity = torch.cat(collated_rbf_similarity)

        mean_distance = torch.mean(collated_distances)
        mean_rbf_similarity = torch.mean(collated_rbf_similarity)

        wandb.log({self.log_distance_name: mean_distance.item()})
        wandb.log({self.log_rbf_similarity_name:mean_rbf_similarity})
    '''
    def on_validation_epoch_end(self,trainer,pl_module):
        optimal_centroids = self.optimal_centroids(pl_module)
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        collated_distances = []
        collated_rbf_similarity = []
        for step, (data,labels,indices) in enumerate(loader):
            assert len(loader) >0, 'dataloader is empty'
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
                data, labels =  data.to(pl_module.device), labels.to(pl_module.device)
                latent_vector = pl_module.callback_vector(data)

            distances = self.distance(pl_module,latent_vector,optimal_centroids)
            #import ipdb;ipdb.set_trace()
            labels = labels.reshape(len(labels),1)
            # gather takes the values from distances along dimension 1 based on the values of labels
            class_distances = torch.gather(distances,1,labels)
            class_rbf_sim = self.rbf_similarity(class_distances)

            collated_distances.append(class_distances)
            collated_rbf_similarity.append(class_rbf_sim)

        collated_distances = torch.cat(collated_distances) # combine all the values
        collated_rbf_similarity = torch.cat(collated_rbf_similarity)

        mean_distance = torch.mean(collated_distances)
        mean_rbf_similarity = torch.mean(collated_rbf_similarity)

        wandb.log({self.log_distance_name: mean_distance.item()})
        wandb.log({self.log_rbf_similarity_name:mean_rbf_similarity})

        #import ipdb; ipdb.set_trace()
        

    def optimal_centroids(self,pl_module):
        centroids_list = []
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        for step, (data,labels,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(data,tuple) or isinstance(data,list):
                data, *aug_data = data
            data,labels = data.to(pl_module.device), labels.to(pl_module.device)

            centroids = self.update_embeddings(pl_module, data, labels)
            centroids_list.append(centroids)
        
        collated_centroids = torch.stack(centroids_list)
        optimal_centroids = torch.mean(collated_centroids, dim=0)
        return optimal_centroids

    @torch.no_grad()
    def update_embeddings(self, pl_module, x, labels): # Assume y is one hot encoder
        z = pl_module.callback_vector(x) # (batch,features)
        y = F.one_hot(labels,num_classes = pl_module.num_classes).float()
        # compute sum of embeddings on class by class basis

        #features_sum = torch.einsum('ij,ik->kj',z,y) # (batch, features) , (batch, num classes) to get (num classes,features)
        #y = y.float() # Need to change into a float to be able to use it for the matrix multiplication
        #features_sum = torch.matmul(y.T,z) # (num_classes,batch) (batch,features) to get (num_class, features)

        features_sum = torch.matmul(z.T, y) # (batch,features) (batch,num_classes) to get (features,num_classes)
        embeddings = features_sum / y.sum(0)
        #embeddings = features_sum.T / y.sum(0) # Nawid - divide each of the feature sum by the number of instances present in the class (need to transpose to get into shape which can divide column wise) shape : (features,num_classes)
        embeddings = embeddings.T # Turn back into shape (num_classes,features)
        return embeddings

    def distance(self,pl_module, x, centroids):
        n = x.size(0) # (batch,features)
        m = centroids.size(0) # (num classes, features)
        d = x.size(1)
        
        assert d == centroids.size(1)

        x = x.unsqueeze(1).expand(n, m, d) # (batch,num_classes, features)
        centroids = centroids.unsqueeze(0).expand(n, m, d) # (batch,num_classes,features)
        diff = x - centroids # (batch,num_classes,features) 
        distances = diff.sum(2) # (batch,num_classes) - distances to each class centroid for each data point in the batch
        #distances = -torch.pow(diff,2).sum(2) # Need to get the negative distance
        return distances

    def rbf_similarity(self,distances):
        exp_similarity = (-(distances**2)).div(2*1**2).exp() # square the distances and divide by a scaling temr
        return exp_similarity
    

        #confidence, indices = torch.max(y_pred,dim=1)
        
    #\n",    
    "        confidence,indices =  confidence.reshape(len(confidence),1), indices.reshape(len(indices),1) # reshape the tensors\n",
    "        density_targets = torch.zeros(len(confidence),2).to(self.device)\n",
    "        density_targets.scatter(1,indices,confidence) # place the values of confidences in the locations specified by indices\n",






class SupConLoss(pl.Callback):
    def __init__(self,datamodule,quick_callback, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.datamodule = datamodule
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.quick_callback = quick_callback
        self.log_name = 'SupCon' 
    
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        dataloader  = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, dataloader)
        for i, (imgs, labels) in enumerate(loader):
            # Obtain the image representations of the data for the specific task
            # Concatenate the two different views of the same images
            imgs = torch.cat([imgs[0],imgs[1]],dim=0) 
            bsz = labels.shape[0]

            imgs,labels = imgs.to(pl_module.device), labels.to(pl_module.device)

            features = pl_module.feature_vector(imgs)
            features_q, features_k = torch.split(features, [bsz,bsz],dim=0)
            features = torch.cat([features_q.unsqueeze(1),features_k.unsqueeze(1)],dim=1)
            self.forward(pl_module,features,labels)
    '''

    # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
    # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
    def forward(self,pl_module, features, labels=None, mask=None):
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
            mask = torch.eye(batch_size, dtype=torch.float32, device = pl_module.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(pl_module.device)
        else:
            mask = mask.float().to(pl_module.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # Nawid - anchor is only the index itself and only the single view
            anchor_count = 1 # Nawid - only one anchor
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature 
            anchor_count = contrast_count # Nawid - all the different views are the anchors
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div( # Nawid - similarity between the anchor and the contrast feature
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(pl_module.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        wandb.log({self.log_name: loss.item()})

        return loss

# choose whether to iterate over the entire dataset or over a single batch of the data (https://github.com/pytorch/pytorch/issues/1917 - obtaining a single batch)
def quickloading(quick_test,dataloader):
    if quick_test:
        loader = [next(iter(dataloader))] # This obtains a single batch of the data as a list which can be iterated
    else:
        loader = dataloader
    return loader
