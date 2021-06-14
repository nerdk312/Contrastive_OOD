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

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading

from scipy.spatial.distance import cdist # Required for TwoMoons visualisation involving pairwise distances
from PIL import Image
from torchvision.utils import save_image



class Variational(pl.Callback): # General class for visualisation
    def __init__(self, datamodule, 
        vector_level:str = 'instance',
        label_level:str = 'fine',
        quick_callback:bool = True):

        self.datamodule = datamodule
        self.datamodule.setup()
        self.quick_callback = quick_callback

        self.vector_level = vector_level
        self.label_level = label_level
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer,pl_module)


    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer, pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1}} 

        train_loader = self.datamodule.train_dataloader()
        features_train, labels_train = self.get_features(pl_module, train_loader)
        class_means, class_examples = self.get_class_means(features_train,labels_train)
        class_means, class_examples = torch.from_numpy(class_means).to(pl_module.device),torch.from_numpy(class_examples).to(pl_module.device)
        reconstructed_class_means = pl_module.decode(class_means)
        reconstructed_class_examples = pl_module.decode(class_examples)
        print('variational callback runs')
        trainer.logger.experiment.log({
                'Mean images': [wandb.Image(x)
                                for x in reconstructed_class_means],
                "global_step": trainer.global_step #pl_module.current_epoch
                })
        
        trainer.logger.experiment.log({
                'class example images': [wandb.Image(x)
                                for x in reconstructed_class_examples],
                "global_step": trainer.global_step #pl_module.current_epoch
                })
        #import ipdb; ipdb.set_trace()

            
    # Obtain the representations for the data
    def get_features(self, pl_module, dataloader):
        features, labels = [], []
        
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label,indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations

            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = self.vector_dict['label_level'][self.label_level]
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]
                         
            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            feature_vector = self.vector_dict['vector_level'][self.vector_level](img) # Performs the callback for the desired level
            features += list(feature_vector.data.cpu().numpy())
            labels += list(label.data.cpu().numpy())
            
        # shape retured is (batch_size, embedding size)
        
        return np.array(features), np.array(labels)
    
    def get_class_means(self, ftrain, ypred):
        #import ipdb; ipdb.set_trace()
        # Nawid - get all the features which belong to each of the different classes
        # Get all the datapoints for the particular case
        xc = [ftrain[ypred == i] for i in np.unique(ypred)] # Nawid - training data which have been predicted to belong to a particular class
        # Find the mean of all the data points in a particular class to get the class means
        class_examples = [x[0:4] for x in xc]
        class_means = [np.mean(x, axis=0) for x in xc]

        # Obtain the class means of the data
        return np.array(class_means), np.vstack(class_examples) # obtain 4 examples from the class for the purpose of evaluating


'''
def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
'''