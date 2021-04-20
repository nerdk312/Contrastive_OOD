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

# Used to log input images and predictions of the dataset

class ImagePredictionLogger(pl.Callback):
    def __init__(self, samples,ood_samples, collated_samples=32):
        super().__init__()
        self.collated_samples = collated_samples # The number of samples to save
        self.imgs, self.labels,self.indices = samples


        self.ood_imgs, self.ood_labels,self.ood_indices = ood_samples
        if isinstance(self.imgs,tuple) or isinstance(self.imgs,list):
            self.imgs ,*aug_imgs  = self.imgs
            self.ood_imgs, *aug_OOD_imgs= self.ood_imgs


    def on_validation_epoch_end(self,trainer,pl_module):
        imgs = self.imgs.to(device=pl_module.device)
        labels = self.labels.to(device=pl_module.device)
        # Plot examples and OOD examples
        self.plot_examples(trainer,pl_module)
        self.plot_OOD_examples(trainer,pl_module)


    def plot_examples(self,trainer,pl_module):
        imgs = self.imgs.to(device=pl_module.device)
        y_pred = class_discrimination(pl_module,imgs) # Nawid - forward pass of all data( can be only training data , or training and noisy data)
    
        preds = torch.argmax(y_pred,-1).cpu() # Get the predictions from the model

        #collated_samples = 16 # number of images desired to use
        for i in range(pl_module.hparams.num_classes): # Iterate through the different classes
            #correct_indices = torch.tensor(np.logical_and(preds==i, self.labels==i),dtype=torch.bool)
            correct_indices = np.logical_and(preds==i, self.labels==i).clone().detach().bool()
            #print('practice indices',practice_indices)
            correct_imgs = imgs[correct_indices]
            num_correct = min(self.collated_samples,len(correct_imgs)) # takes min of the desired amount and the number which is actually correct
            correct_dict_name = 'Class %d Correct' %i

            trainer.logger.experiment.log({
                correct_dict_name: [wandb.Image(x)
                                for x in correct_imgs[:num_correct]],
                "global_step": trainer.global_step #pl_module.current_epoch
                })


            #incorrect_indices = torch.tensor(np.logical_and(preds==i, self.labels!=i),dtype= torch.bool)
            incorrect_indices = np.logical_and(preds==i, self.labels!=i).clone().detach().bool()
            incorrect_imgs = imgs[incorrect_indices]

            #incorrect_imgs = imgs[np.logical_and(preds!=i, self.labels==i)]

            num_incorrect = min(self.collated_samples,len(incorrect_imgs)) # takes min of the desired amount and the number which is actually correct

            incorrect_dict_name = 'Class %d Incorrect' %i

            trainer.logger.experiment.log({
                incorrect_dict_name: [wandb.Image(x,caption=f"Pred:{pred}, Label:{y}")
                                for x,pred,y in zip(incorrect_imgs[:num_incorrect],preds[incorrect_indices][:num_incorrect],self.labels[incorrect_indices][:num_incorrect])],
                "global_step": trainer.global_step #pl_module.current_epoch
                })
            # Gets incorrect imgs from 0 to num_incorrect, gets the preds indices where the number was incorrect and gets only the first num_incorrect values of them (same thing with the actual labels)

    def plot_OOD_examples(self,trainer,pl_module):

        ood_imgs = self.ood_imgs.to(device=pl_module.device)
        ood_labels = self.ood_labels.to(device=pl_module.device)
        y_pred = class_discrimination(pl_module,ood_imgs)
        #_,y_pred = pl_module.online_encoder(ood_imgs,centroids) # Nawid - forward pass of all data( can be only training data , or training and noisy data)
        # Obtain the max scores
        confidence, preds = torch.max(y_pred,dim=1) # predictions and max scores
        sorted_confidences, sort_indices = torch.sort(confidence) # Sort the confidence values from high to low # COULD ALSO USE TORCH.TOPK TO PERFORM THIS https://pytorch.org/docs/stable/generated/torch.topk.html

        # Order the indices from largest to smallest (in terms of the batch dimension)
        sorted_ood_imgs = ood_imgs[sort_indices]
        sorted_labels = ood_labels[sort_indices]
        sorted_preds = preds[sort_indices]

        # Go through loop for the different classes and see which class is being predicted by each model

        #collated_samples = 16 # number of images desired to use
        for i in range(pl_module.hparams.num_classes): # Iterate through the different classes
            ood_sorted_class_i = sorted_ood_imgs[sorted_preds == i] # Get the sorted class i images based on the sorted predictions being equal to i

            num_incorrect = min(self.collated_samples,len(ood_sorted_class_i)) # takes min of the desired amount and the number which is actually correct
            incorrect_dict_name = 'OOD Class %d Incorrect' %i

            trainer.logger.experiment.log({
                incorrect_dict_name: [wandb.Image(x,caption=f"Pred:{pred}, Label:{y}")
                                for x,pred,y in zip(ood_sorted_class_i[:num_incorrect],sorted_preds[sorted_preds == i][:num_incorrect],sorted_labels[sorted_preds == i][:num_incorrect])],
                "global_step": trainer.global_step
                })


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
            for data, target in loader:
                if isinstance(data, tuple) or isinstance(data, list):
                    data, *aug_data = data # Used to take into accoutn whether the data is a tuple of the different augmentations

                data = data.to(pl_module.device)

                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = pl_module.feature_vector(data) # representation of the data
                
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
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            img = img.to(pl_module.device)
            features.append(pl_module.feature_vector(img))
        
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

        #import ipdb; ipdb.set_trace()
        

    def optimal_centroids(self,pl_module):
        centroids_list = []
        test_loader = self.datamodule.test_dataloader()
        loader = quickloading(self.quick_callback, test_loader)
        for step, (data,labels) in enumerate(loader):
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
        z = pl_module.feature_vector(x) # (batch,features)
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



class OOD_ROC(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule

        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        #self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

    # Outputs the OOD scores only (does not output the curve)
    def on_validation_epoch_end(self,trainer,pl_module):
        #print('OOD ROC training?:',pl_module.encoder_q.training)
        accuracy, roc_auc = self.get_auroc_ood_basic(pl_module)

    # Outputs OOD scores aswell as the ROC curve
    def on_test_epoch_end(self,trainer,pl_module):
        #print('OOD ROC training?:',pl_module.encoder_q.training)
        accuracy, roc_auc = self.get_auroc_ood(pl_module)


    def prepare_ood_datasets(self): # Code seems to be same as DUQ implementation
        true_dataset,ood_dataset = self.Datamodule.test_dataset, self.OOD_Datamodule.test_dataset # Dataset is not transformed at this point
        datasets = [true_dataset, ood_dataset]

        # Nawid - targets for anomaly, 0 is not anomaly and 1 is anomaly
        anomaly_targets = torch.cat(
            (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
        )

        concat_datasets = torch.utils.data.ConcatDataset(datasets)
        # Dataset becomes transformed as it passes through the datalodaer I believe
        dataloader = torch.utils.data.DataLoader(
            concat_datasets, batch_size=100, shuffle=False, num_workers=6, pin_memory=False,
        )

        return dataloader, anomaly_targets # Nawid -contains the data for the true data and the false data aswell as values which indicate whether it is an anomaly or not

    def loop_over_dataloader(self,pl_module, dataloader): # Nawid - obtain accuracy (pred equal to target) and kernel distances which are the scores
        #model.eval()

        with torch.no_grad():
            scores = []
            accuracies = []
            outputs = []
            #loader = quickloading(self.quick_callback,dataloader) # Used to get a single batch or used to get the entire dataset
            for data, target,indices in dataloader:
                if isinstance(data, tuple) or isinstance(data, list):
                    data, *aug_data = data # Used to take into accoutn whether the data is a tuple of the different augmentations

                data = data.to(pl_module.device)
                target = target.to(pl_module.device)

                # Nawid - distance which is the score as well as the prediciton for the accuracy is obtained
                output = class_discrimination(pl_module,data)
                kernel_distance, pred = output.max(1) # The higher the kernel distance is, the more confident it is, so the lower the probability of an outlier

                # Make array for the probability of being outlier and not being an outlier which is based on the confidence (kernel distance)
                OOD_prediction = torch.zeros((len(data),2)).to(pl_module.device)

                OOD_prediction[:,0] = kernel_distance # The kernel distance is the confidence that the label is 0 (not outlier)
                OOD_prediction[:,1] = 1 - OOD_prediction[:,0] # Value of being an outlier (belonging to class 1), is 1 - kernel distance (1-confidence)

                accuracy = pred.eq(target)
                accuracies.append(accuracy.cpu().numpy())

                scores.append(-kernel_distance.cpu().numpy()) # Scores represent - kernel distance, since kernel distance is the confidence of being in domain, - kernel distance represent OOD samples
                outputs.append(OOD_prediction.cpu().numpy())

        scores = np.concatenate(scores)
        accuracies = np.concatenate(accuracies)
        outputs = np.concatenate(outputs)

        return scores, accuracies,outputs

    # Outputs the OOD values as well as the ROC curve
    def get_auroc_ood(self,pl_module):
        dataloader, anomaly_targets = self.prepare_ood_datasets() # Nawid - obtains concatenated true and false data, as well as anomaly targets which show if the value is 0 (not anomaly) or 1 (anomaly)

        scores, accuracies,outputs = self.loop_over_dataloader(pl_module, dataloader)

        confidence_scores = outputs[:,0]
        # confidence scores for the true data and for the OOD data
        true_scores =  confidence_scores[: len(self.Datamodule.test_dataset)]
        ood_scores =  confidence_scores[len(self.Datamodule.test_dataset):]

        # histogram of the confidence scores for the true data
        true_data = [[s] for s in true_scores]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({'True_data_OOD_scores': wandb.plot.histogram(true_table, "scores",title="true_data_scores")})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in ood_scores]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({'OOD_data_OOD_scores': wandb.plot.histogram(ood_table, "scores",title="OOD_data_scores")})

        
        # Double checking the calculation of the ROC scores for the data
        '''
        second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        wandb.log({"AUROC_v2":second_check_roc_auc})
        '''

        accuracy = np.mean(accuracies[: len(self.Datamodule.test_dataset)]) # Nawid - obtainn accuracy for the true dataset
        roc_auc = roc_auc_score(anomaly_targets, scores) # Nawid - obtain roc auc for a binary classification problem where the scores show how close it is to a centroid (which will say whether it is in domain or out of domain), and whether it identified as being in domain or out of domain.

        wandb.log({"roc" : wandb.plot.roc_curve(anomaly_targets, outputs,#scores,
                        labels=None, classes_to_plot=None)})

        wandb.log({"AUROC":roc_auc})

        return accuracy, roc_auc

    
    def get_auroc_ood_basic(self,pl_module):
        dataloader, anomaly_targets = self.prepare_ood_datasets() # Nawid - obtains concatenated true and false data, as well as anomaly targets which show if the value is 0 (not anomaly) or 1 (anomaly)

        scores, accuracies,outputs = self.loop_over_dataloader(pl_module, dataloader)

        confidence_scores = outputs[:,0]
        # confidence scores for the true data and for the OOD data
        true_scores =  confidence_scores[: len(self.Datamodule.test_dataset)]
        ood_scores =  confidence_scores[len(self.Datamodule.test_dataset):]

        # histogram of the confidence scores for the true data
        true_data = [[s] for s in true_scores]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({'True_data_OOD_scores': wandb.plot.histogram(true_table, "scores",title="true_data_scores")})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in ood_scores]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({'OOD_data_OOD_scores': wandb.plot.histogram(ood_table, "scores",title="OOD_data_scores")})

        '''
        # Double checking the calculation of the ROC scores for the data
        #second_check_scores = #1 + scores # scores are all negative values
        second_check_roc_auc = roc_auc_score(1-anomaly_targets,confidence_scores) # ROC scores for the case of checking whether a sample is in distribution data
        wandb.log({"AUROC_v2":second_check_roc_auc})
        '''

        accuracy = np.mean(accuracies[: len(self.Datamodule.test_dataset)]) # Nawid - obtainn accuracy for the true dataset
        roc_auc = roc_auc_score(anomaly_targets, scores) # Nawid - obtain roc auc for a binary classification problem where the scores show how close it is to a centroid (which will say whether it is in domain or out of domain), and whether it identified as being in domain or out of domain.

        wandb.log({"AUROC":roc_auc})

        return accuracy, roc_auc


class Mahalanobis_OOD(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule,quick_callback):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        self.log_name = "Mahalanobis_"
        self.true_histogram = 'Mahalanobis_True_data_scores'
        self.ood_histogram = 'Mahalanobis_OOD_data_scores'

        # Names for creating a confusion matrix for the data
        class_dict = self.Datamodule.idx2class
        self.class_names = [v for k,v in class_dict.items()] # names of the categories of the dataset
        # Obtain class names list for the case of the OOD data
        OOD_class_dict = self.OOD_Datamodule.idx2class
        self.OOD_class_names = [v for k,v in OOD_class_dict.items()] # names of the categories of the dataset
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
            
        features_train, labels_train = self.get_features(pl_module, train_loader)  # using feature befor MLP-head
        features_test, labels_test = self.get_features(pl_module, train_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        # Obtain the fpr95, aupr, test predictions, OOD predictions
        fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
        )
        self.distance_confusion_matrix(trainer,indices_dtest,labels_test)
        self.distance_OOD_confusion_matrix(trainer,indices_dood,labels_ood)

        return fpr95,auroc,aupr 
    '''
    def on_validation_epoch_end(self,trainer,pl_module):
        train_loader = self.Datamodule.train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()
            
        features_train, labels_train = self.get_features(pl_module, train_loader)  # using feature befor MLP-head
        features_test, labels_test = self.get_features(pl_module, train_loader)
        features_ood, labels_ood = self.get_features(pl_module, ood_loader)
        
        # Obtain the fpr95, aupr, test predictions, OOD predictions (basic version does not log the confidence scores or the confusion matrices)
        fpr95, auroc, aupr, indices_dtest, indices_dood = self.get_eval_results_basic(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
        )

        return fpr95,auroc,aupr 

    def get_features(self,pl_module, dataloader, max_images=10**10, verbose=False):
        features, labels = [], []
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, label) in enumerate(loader):
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations


            if total > max_images:
                break
            
            img, label = img.to(pl_module.device), label.to(pl_module.device)

            features += list(pl_module.feature_vector(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            if verbose and not index % 50:
                print(index)
                
            total += len(img)

        return np.array(features), np.array(labels)    
    def get_scores(self,ftrain, ftest, food, labelstrain):
        ypred = labelstrain
        return self.get_scores_multi_cluster(ftrain, ftest, food, ypred)

    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        
        din = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True)) # Nawid - distance between the data point and the mean
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (ftest - np.mean(x, axis=0, keepdims=True)).T
                    ) # Nawid - calculating the covariance matrix of the data belonging to a particular class and dot preduct by the distance of the data point from the mean (distance calculation)
                ).T,
                axis=-1,
            )
            for x in xc # Nawid - done for all the different classes
        ]
        
        dood = [
            np.sum(
                (food - np.mean(x, axis=0, keepdims=True))
                * (
                    np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                        (food - np.mean(x, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
            for x in xc # Nawid- this calculates the score for all the OOD examples 
        ]
        # Calculate the indices corresponding to the values
        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood
    
    # Changes OOD scores to confidence scores 
    def log_confidence_scores(self,Dtest,DOOD):
        '''
        confidence_test = np.exp(-Dtest)
        confidence_OOD  = np.exp(-DOOD)
        '''
        confidence_test = Dtest
        confidence_OOD  = DOOD
         # histogram of the confidence scores for the true data
        true_data = [[s] for s in confidence_test]
        true_table = wandb.Table(data=true_data, columns=["scores"])
        wandb.log({self.true_histogram: wandb.plot.histogram(true_table, "scores",title= self.true_histogram)})

        # Histogram of the confidence scores for the OOD data
        ood_data = [[s] for s in confidence_OOD]
        ood_table = wandb.Table(data=ood_data, columns=["scores"])
        wandb.log({self.ood_histogram: wandb.plot.histogram(ood_table, "scores",title=self.ood_histogram)})
    
    def get_eval_results(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain)
        self.log_confidence_scores(dtest,dood)

        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
        wandb.log({self.log_name + 'AUROC': auroc})
        return fpr95, auroc, aupr, indices_dtest, indices_dood

    def get_eval_results_basic(self,ftrain, ftest, food, labelstrain):
        """
            None.
        """
        # Nawid -normalise the featues for the training, test and ood data
        # standardize data
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
        # Nawid - calculate the mean and std of the traiing features
        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)
        # Nawid - normalise data using the mean and std
        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)
        # Nawid - obtain the scores for the test data and the OOD data
        dtest, dood, indices_dtest, indices_dood = self.get_scores(ftrain, ftest, food, labelstrain)


        # Nawid- get false postive rate and asweel as AUROC and aupr
        fpr95 = get_fpr(dtest, dood)
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
        wandb.log({self.log_name + 'AUROC': auroc})
        return fpr95, auroc, aupr, indices_dtest, indices_dood
        
    
    def distance_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +"conf_mat_id": wandb.plot.confusion_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names),
            "global_step": trainer.global_step
                  })

    def distance_OOD_confusion_matrix(self,trainer,predictions,labels):
        wandb.log({self.log_name +"OOD_conf_mat_id": OOD_conf_matrix(probs = None,
            preds=predictions, y_true=labels,
            class_names=self.class_names,OOD_class_names =self.OOD_class_names),
            "global_step": trainer.global_step
                  })

# Same as mahalanobis but obtains different feature vector for the task
class Mahalanobis_OOD_compressed(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,quick_callback):
        super().__init__(Datamodule,OOD_Datamodule,quick_callback)
        self.log_name = "Mahalanobis_compressed_"    
        self.true_histogram = 'Mahalanobis_True_data_compressed'
        self.ood_histogram = 'Mahalanobis_OOD_data_compressed'

        

    def get_features(self,pl_module, dataloader, max_images=10**10, verbose=False):
        features, labels = [], []
        total = 0
        loader = quickloading(self.quick_callback,dataloader)
        for index, (img, label) in enumerate(loader):
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations


            if total > max_images:
                break
            
            img, label = img.to(pl_module.device), label.to(pl_module.device)

            features += list(pl_module.feature_vector_compressed(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            if verbose and not index % 50:
                print(index)
                
            total += len(img)

        return np.array(features), np.array(labels)
    
# Same as mahalanobis but calculates score based on the minimum euclidean distance rather than min Mahalanobis distance
class Euclidean_OOD(Mahalanobis_OOD):
    def __init__(self, Datamodule,OOD_Datamodule,quick_callback):
        super().__init__(Datamodule,OOD_Datamodule,quick_callback)
        self.log_name = "Euclidean_"
        self.true_histogram = 'Euclidean_True_data_scores'
        self.ood_histogram = 'Euclidean_OOD_data_scores'


    def get_scores_multi_cluster(self,ftrain, ftest, food, ypred):
        # Nawid - get all the features which belong to each of the different classes
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Calculate the sqrt of the sum of the squared distances
        din = [
            np.sqrt(np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True))**2,axis=-1,))
            for x in xc # Nawid - done for all the different classes
        ]
        dood = [
            np.sqrt(np.sum(
                (food - np.mean(x, axis=0, keepdims=True))**2,axis=-1,))
            for x in xc # Nawid - done for all the different classes
        ]

        indices_din = np.argmin(din,axis = 0)
        indices_dood = np.argmin(dood, axis=0)

        din = np.min(din, axis=0) # Nawid - calculate the minimum distance 
        dood = np.min(dood, axis=0)

        return din, dood, indices_din, indices_dood




class OOD_confusion_matrix(pl.Callback):
    def __init__(self, Datamodule,OOD_Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        #import ipdb; ipdb.set_trace()
        # Obtain class names list for the case of the indomain data
        class_dict = Datamodule.idx2class

        self.class_names = [v for k,v in class_dict.items()] # names of the categories of the dataset
        
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.OOD_Datamodule.setup() # Set up again to reset the after providing the transform

        # Obtain class names list for the case of the OOD data
        OOD_class_dict = OOD_Datamodule.idx2class

        #OOD_class_dict = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five", 6:"six",7:"seven",8:"eight",9:"nine"}

        self.OOD_class_names = [v for k,v in OOD_class_dict.items()] # names of the categories of the dataset

    # Obtain confusion matrix at the test epoch end
    def on_test_epoch_end(self,trainer,pl_module):
        #print('OOD confusion matrix training?:',pl_module.encoder_q.training)
        pl_module.eval() # change to evaluation mode
        
        #optimal_centroids = self.optimal_centroids(trainer,pl_module)
        self.OOD_confusion(trainer,pl_module)


    def OOD_confusion(self,trainer,pl_module):
        OOD_test_dataset = self.OOD_Datamodule.test_dataset

        OOD_test_dataloader = torch.utils.data.DataLoader(
            OOD_test_dataset, batch_size=self.Datamodule.batch_size, shuffle=False, num_workers=6, pin_memory=False,
        )
        OOD_preds = []
        for (OOD_data, _) in OOD_test_dataloader:
            if isinstance(OOD_data, tuple) or isinstance(OOD_data, list):
                OOD_data, *aug_data = OOD_data # Used to take into accoutn whether the data is a tuple of the different augmentations

            OOD_data = OOD_data.to(pl_module.device)
            y_pred = class_discrimination(pl_module,OOD_data)

            OOD_preds.append(y_pred)

        OOD_preds = torch.cat(OOD_preds) #  Predictions on the actual data where the targets are the indomain centroids

        OOD_targets = OOD_test_dataset.targets# if hasattr(OOD_test_dataset, 'targets') else torch.from_numpy(OOD_test_dataset.labels) # Targets based on the OOD data (targets or labels used for the different situations)

        top_pred_ids = OOD_preds.argmax(axis=1)
        

        wandb.log({"my_OOD_conf_mat_id" : OOD_conf_matrix(probs = OOD_preds.cpu().numpy(),
            preds=None, y_true=OOD_targets.cpu().numpy(),
            class_names=self.class_names,OOD_class_names =self.OOD_class_names),
            "global_step": trainer.global_step
                  })



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


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

# calculates aupr
def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin)  + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr

# Nawid - calculate false positive rate
def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

# choose whether to iterate over the entire dataset or over a single batch of the data (https://github.com/pytorch/pytorch/issues/1917 - obtaining a single batch)
def quickloading(quick_test,dataloader):
    if quick_test:
        loader = [next(iter(dataloader))] # This obtains a single batch of the data as a list which can be iterated
    else:
        loader = dataloader
    return loader
