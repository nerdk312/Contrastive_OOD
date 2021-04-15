import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import sklearn.metrics as skm
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from Contrastive_uncertainty.SupConPCL.callbacks.general_callbacks import quickloading



from scipy.spatial.distance import cdist # Required for TwoMoons visualisation involving pairwise distances

class Visualisation(pl.Callback): # General class for visualisation
    def __init__(self, datamodule, ood_datamodule, quick_callback):
        self.datamodule = datamodule
        self.ood_datamodule = ood_datamodule
        self.ood_datamodule.test_transforms= self.datamodule.test_transforms   # Make it so that the OOD datamodule has the same transform as the true module

        self.datamodule.setup()
        self.ood_datamodule.setup()
        # setup data
        self.quick_callback = quick_callback
    
    def on_test_epoch_end(self, trainer, pl_module):
        representations, labels = self.obtain_representations(pl_module)
        #concat_representations, concat_labels, class_concat_labels = self..OOD_representations() # obtain representations and labels which have both data

        # +1 in num classes to represent outlier data, *2 represents class specific outliers
        self.pca_visualisation(representations, labels, pl_module.hparams.num_classes, 'inliers')
        #self.pca_visualisation(concat_representations, concat_labels,config['num_classes']+1,'general')
        #self.pca_visualisation(concat_representations, class_concat_labels,2* config['num_classes'],'class')

        self.tsne_visualisation(representations, labels, pl_module.hparams.num_classes, 'inliers')

    def obtain_representations(self, pl_module): #  separate from init so that two moons does not make representations automatically using dataloader rather it uses X_vis
        # self.data = self.datamodule.test_dataloader() # instantiate val dataloader

        self.data = self.datamodule.test_dataset
        dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=200, shuffle=False, num_workers=6, pin_memory=False
            ) # Dataloader with batch size 500 to ensure that all the data is obtained for the tas
        
        #self.labels = self.datamodule.test_dataset.targets  # Obtain the test dataset targets

        loader = quickloading(self.quick_callback, dataloader)
        self.representations, self.labels = self.compute_representations(pl_module, loader)
        return self.representations, self.labels

    def OOD_representations(self):
        true_dataset = self.datamodule.test_dataset  #  Test set of the true datamodule

        #ood_datamodule.test_dataset.targets  =  -(ood_datamodule.test_dataset.targets +1)# represent the OOD class with negative values
        ood_dataset = self.ood_datamodule.test_dataset
        datasets = [true_dataset, ood_dataset]

        # General level OOD labels and class specific OOD labels
        concat_labels = torch.cat([true_dataset.targets, torch.zeros_like(ood_datamodule.test_dataset.targets).fill_(-1)]) # update the targets to be values of -1 to represent the anomaly that they are anomalous values
        class_concat_labels = torch.cat([true_dataset.targets, -(ood_datamodule.test_dataset.targets +1)]) # Increase values by 1 to get values 1 to 10 and then change negative to prevent overlapping with the real class labels

        concat_datasets = torch.utils.data.ConcatDataset(datasets)

        dataloader = torch.utils.data.DataLoader(
                concat_datasets, batch_size=200, shuffle=False, num_workers=6, pin_memory=False
            )
        loader = quickloading(self.quick_callback, dataloader)

        self.concat_representations = self.compute_representations(loader)
        return self.concat_representations, concat_labels, class_concat_labels

    @torch.no_grad()
    def compute_representations(self, pl_module, loader):
        features = []
        collated_labels = []
        for i, (images, labels, indices) in enumerate(tqdm(loader)): # Obtain data and labels from dataloader
            assert len(loader)>0, 'loader is empty'
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_imgs = images
            
            images = images.to(pl_module.device)  # cuda(non_blocking=True)
            features.append(pl_module.callback_vector(images))  # Obtain features
            collated_labels.append(labels)

        features = torch.cat(features)
        collated_labels = torch.cat(collated_labels)
        return features.cpu(), collated_labels.cpu()


    def pca_visualisation(self, representations, labels, num_classes, name):
        pca = PCA(n_components=3)

        pca_result = pca.fit_transform(representations)
        pca_one = pca_result[:,0]
        pca_two = pca_result[:,1]
        pca_three = pca_result[:,2]

        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=pca_one, y=pca_two,
        hue = labels,
        palette=sns.color_palette("hls",num_classes),
        legend="full",
        alpha=0.3
        )
        # Limits for te plot

        #sns.plt.xlim(-2.5,2.5)
        #sns.plt.ylim(-2.5,2.5)

        pca_filename = 'Images/'+name + '_data_pca.png'
        plt.savefig(pca_filename)
        wandb_pca = name +' PCA of Features'
        wandb.log({wandb_pca:wandb.Image(pca_filename)})

        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
        xs=pca_one,
        ys=pca_two,
        zs=pca_three,
        c=labels,
        cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        #limits for plot to ensure fixed scale
        #ax.xlim(-2.5,2.5)
        #ax.ylim(-2.5,2.5)
        pca_3D_filename = 'Images/'+name + ' data_pca_3D.png'
        plt.savefig(pca_3D_filename)
        wandb_3D_pca = name + ' 3D PCA of Features'
        wandb.log({wandb_3D_pca: wandb.Image(pca_3D_filename)})
        plt.close()

    def tsne_visualisation(self, representations, labels, num_classes, name):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(representations)
        tsne_one = tsne_results[:,0]
        tsne_two = tsne_results[:,1]

        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=tsne_one, y=tsne_two,
        hue=labels,
        palette=sns.color_palette("hls", num_classes),
        legend="full",
        alpha=0.3
        )
        # Used to control the scale of a seaborn plot
        #sns.plt.ylim(-15, 15)
        #sns.plt.xlim(-15, 15)

        tsne_filename = 'Images/'+ name +'_data_tsne.png'
        plt.savefig(tsne_filename)
        wandb_tsne = name +' TSNE of Features'
        wandb.log({wandb_tsne:wandb.Image(tsne_filename)})
        plt.close()
