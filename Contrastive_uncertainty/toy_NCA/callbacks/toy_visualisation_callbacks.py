import os
import subprocess
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

import wandb
from sklearn.metrics import roc_auc_score
import sklearn.datasets


import glob
from Contrastive_uncertainty.Moco.moco_callbacks import quickloading, \
                                                         get_fpr, get_pr_sklearn, get_roc_sklearn

from Contrastive_uncertainty.toy_example.callbacks.toy_visualisation_callbacks import circular_visualisation, data_visualisation, \
                                    TwoMoonsUncertaintyVisualisation, TwoMoonsUncertaintyVisualisation

        
'''
class UncertaintyVisualisation(pl.Callback):  # contains methods specifc for the two moons dataset
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule
        self.datamodule.setup()
        # Obtain data for visualisation
        self.x_vis, self.y_vis = self.datamodule.train_data, self.datamodule.train_labels
        
        #self.X_vis, self.y_vis = sklearn.datasets.make_moons(n_samples=2500, noise=self.datamodule.noise) # Nawid - moon dataset
        #self.X_vis = (self.X_vis - self.datamodule.mean)/self.datamodule.std # normalise data

    def on_validation_epoch_end(self,trainer, pl_module):
        self.visualise_uncertainty(pl_module)
    

    def outlier_grid(self): #  Generates the grid of points, outputs, x_lin and y_lin aswell as this is required for the uncertainty visualisation
        domain = 4
        x_lin, y_lin = np.linspace(-domain+0.5, domain+0.5, 50), np.linspace(-domain, domain, 50)

        # Normalising the data which is used for the visualisation
        #x_lin,y_lin = (x_lin -self.datamodule.mean[0])/self.datamodule.std[0], (y_lin -self.datamodule.mean[1])/self.datamodule.std[1]
        xx, yy = np.meshgrid(x_lin, y_lin)
        x_grid = np.column_stack([xx.flatten(), yy.flatten()])
        return x_lin, y_lin, x_grid # Not normalising since a range between -3 and 3 should capture all the data sufficiently for the normalised data

    @torch.no_grad()
    def visualise_uncertainty(self,pl_module):
        # Generates test outlier data
        x_lin, y_lin, x_grid = self.outlier_grid()

        mask = self.y_vis.astype(np.bool)
        centroids = pl_module.update_embeddings(torch.from_numpy(self.x_vis).float().to(pl_module.device),torch.from_numpy(self.y_vis).to(pl_module.device))

        with torch.no_grad():
            output = pl_module(torch.from_numpy(x_grid).float().to(pl_module.device),centroids)
            #import ipdb; ipdb.set_trace()
            confidence = output.max(1)[0].cpu().numpy()

        #z = confidence.reshape(xx.shape) # Original version, replaced with x_lin shape[0] since I placed xx in a function
        z = confidence.reshape((x_lin.shape[0], x_lin.shape[0]))
        plt.figure()
        plt.contourf(x_lin, y_lin, z, cmap='cividis')
        plt.colorbar().set_label('Confidence')

        plt.scatter(self.x_vis[mask,0], self.x_vis[mask,1])
        plt.scatter(self.x_vis[~mask,0], self.x_vis[~mask,1])

        uncertainty_filename = 'Images/uncertainty.png'
        plt.savefig(uncertainty_filename)
        wandb_uncertainty = 'uncertainty'
        wandb.log({wandb_uncertainty:wandb.Image(uncertainty_filename)})
        plt.close()
    
    def animate_uncertainty(self):
        fig = plt.figure()
        ani = animation.ArtistAnimation(fig, self.frames, interval=50, blit=True,
                                repeat_delay=1000)
        plt.show()
'''




