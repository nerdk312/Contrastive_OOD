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

class Visualisation(pl.Callback): # General class for visualisation
    def __init__(self, datamodule, ood_datamodule,quick_callback):
        self.datamodule = datamodule
        self.ood_datamodule = ood_datamodule
        self.ood_datamodule.test_transforms= self.datamodule.test_transforms   # Make it so that the OOD datamodule has the same transform as the true module

        self.datamodule.setup()
        self.ood_datamodule.setup()
        # setup data
        self.quick_callback = quick_callback

    #def on_fit_start(self, trainer, pl_module):
    def on_test_epoch_end(self, trainer, pl_module):
        # Obtain representations for the normal case as well as the concatenated representations
        
        representations, labels = self.obtain_representations(pl_module)
        
        # PCA visalisation
        self.pca_visualisation(representations, labels, 'inliers')
        # T-SNE visualisation
        self.tsne_visualisation(representations, labels, 'inliers')
        # +1 in num classes to represent outlier data, *2 represents class specific outliers
        #self.pca_visualisation(concat_representations, concat_labels,config['num_classes']+1,'general')

        # Obtain the concatenated representations for the case where the different values can be used for the task
        if not self.quick_callback:
            concat_representations, concat_labels, class_concat_labels = self.OOD_representations(pl_module) # obtain representations and labels which have both data
            self.pca_visualisation(concat_representations, class_concat_labels, 'class')
            self.tsne_visualisation(concat_representations, class_concat_labels, 'class')

    def obtain_representations(self, pl_module):  # separate from init so that two moons does not make representations automatically using dataloader rather it uses X_vis
        # self.data = self.datamodule.test_dataloader() # instantiate val dataloader

        self.data = self.datamodule.test_dataset
        dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=200, shuffle=False, num_workers=6, pin_memory=False
            ) # Dataloader with batch size 500 to ensure that all the data is obtained for the tas
        
        #self.labels = self.datamodule.test_dataset.targets  # Obtain the test dataset targets

        loader = quickloading(self.quick_callback, dataloader)
        self.representations, self.labels = self.compute_representations(pl_module, loader)
        return self.representations, self.labels

    def OOD_representations(self,pl_module):
        true_dataset = self.datamodule.test_dataset  #  Test set of the true datamodule

        #ood_datamodule.test_dataset.targets  =  -(ood_datamodule.test_dataset.targets +1)# represent the OOD class with negative values
        ood_dataset = self.ood_datamodule.test_dataset
        datasets = [true_dataset, ood_dataset]

        # General level OOD labels and class specific OOD labels
        concat_labels = torch.cat([true_dataset.targets, torch.zeros_like(self.ood_datamodule.test_dataset.targets).fill_(-1)]) # update the targets to be values of -1 to represent the anomaly that they are anomalous values
        class_concat_labels = torch.cat([true_dataset.targets, -(self.ood_datamodule.test_dataset.targets +1)]) # Increase values by 1 to get values 1 to 10 and then change negative to prevent overlapping with the real class labels

        concat_datasets = torch.utils.data.ConcatDataset(datasets)

        dataloader = torch.utils.data.DataLoader(
                concat_datasets, batch_size=200, shuffle=False, num_workers=6, pin_memory=False
            )
        loader = quickloading(self.quick_callback, dataloader)

        self.concat_representations, _ = self.compute_representations(pl_module, loader)
        return self.concat_representations, concat_labels, class_concat_labels

    @torch.no_grad()
    def compute_representations(self, pl_module, loader):
        features = []
        collated_labels = []
        #import ipdb; ipdb.set_trace()
        for i, (images, labels,indices) in enumerate(tqdm(loader)): # Obtain data and labels from dataloader
            assert len(loader)>0, 'loader is empty'
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_imgs = images
            
            images = images.to(pl_module.device)  # cuda(non_blocking=True)
            features.append(pl_module.callback_vector(images))  # Obtain features
            collated_labels.append(labels)

        features = torch.cat(features)
        collated_labels = torch.cat(collated_labels)
        return features.cpu(), collated_labels.cpu()


    def pca_visualisation(self, representations, labels, name):
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(representations)
        pca_one = pca_result[:,0]
        pca_two = pca_result[:,1]
        pca_three = pca_result[:,2]
        
        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=pca_one, y=pca_two,
        hue = labels,
        #palette=sns.color_palette('hls',n_colors =10),
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

    def tsne_visualisation(self, representations, labels, name):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(representations)
        tsne_one = tsne_results[:,0]
        tsne_two = tsne_results[:,1]

        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=tsne_one, y=tsne_two,
        hue=labels,
        #palette=sns.color_palette("hls", num_classes),
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


class circular_visualisation(pl.Callback):
    def __init__(self, Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        #import ipdb; ipdb.set_trace()
        # Obtain class names list for the case of the indomain data

    def on_validation_epoch_end(self, trainer, pl_module):
        collated_features = []
        collated_labels = []
        dataloader = self.Datamodule.train_dataloader()
        for data, labels, indices in dataloader:
            if isinstance(data, tuple) or isinstance(data, list):
                data, *aug_data = data
            data = data.to(pl_module.device)
            feature_vector = pl_module.feature_vector(data)
            collated_features.append(feature_vector)

            collated_labels.append(labels)
            

        collated_features = torch.cat(collated_features)
        collated_features = collated_features.cpu().numpy()

        collated_labels = torch.cat(collated_labels)
        collated_labels = collated_labels.numpy()

        #import ipdb; ipdb.set_trace()

        theta = np.radians(np.linspace(0,360,300))
        x_2 = np.cos(theta)
        y_2 = np.sin(theta)


        plt.plot(x_2, y_2, '--', color='gray', label='Unit Circle')
        for i in range(4):
            loc = np.where(collated_labels==i)[0] # Get all the datapoints of a specific class
            plt.scatter(collated_features[loc,0], collated_features[loc,1])#, color=list(colors[loc,:]), s=60) # plot the data points for a specific class
            plt.savefig('unitnorm.png')
        #ax[1].scatter(x_base_t1[:,0], x_base_t1[:,1], marker='x', color='r', s=60) # Plot the data for the embedding of the x test line
        #ax[1].scatter(x_base_t2[:,0], x_base_t2[:,1], marker='x', color='black', s=60)
        #ax[1].scatter(y_base_t1[:,0], y_base_t1[:,1], marker='^', color='brown', s=60)
        #ax[1].scatter(y_base_t2[:,0], y_base_t2[:,1], marker='^', color='magenta', s=60)
        #ax[1].set_xlim([np.min(base_embed[:,0])*0.85,np.max(base_embed[:,0]*1.15)]) # Set an x limit for the graph
        #ax[1].set_ylim([np.min(base_embed[:,1])*1.15,np.max(base_embed[:,1]*0.85)]) # Set y limit for the graph

        '''    
        # Plotting the second image
        ax[1].plot(x_2, y_2, '--', color='gray', label='Unit Circle')
        for i in range(len(lines)):
            loc = np.where(labels==i)[0] # Get all the datapoints of a specific class
            ax[1].scatter(base_embed[loc,0], base_embed[loc,1], color=list(colors[loc,:]), s=60) # plot the data points for a specific class
        ax[1].scatter(x_base_t1[:,0], x_base_t1[:,1], marker='x', color='r', s=60) # Plot the data for the embedding of the x test line
        ax[1].scatter(x_base_t2[:,0], x_base_t2[:,1], marker='x', color='black', s=60)
        ax[1].scatter(y_base_t1[:,0], y_base_t1[:,1], marker='^', color='brown', s=60)
        ax[1].scatter(y_base_t2[:,0], y_base_t2[:,1], marker='^', color='magenta', s=60)
        ax[1].set_xlim([np.min(base_embed[:,0])*0.85,np.max(base_embed[:,0]*1.15)]) # Set an x limit for the graph
        ax[1].set_ylim([np.min(base_embed[:,1])*1.15,np.max(base_embed[:,1]*0.85)]) # Set y limit for the graph
        '''

class data_visualisation(pl.Callback): 
    def __init__(self, Datamodule, OOD_Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule

        self.Datamodule.setup()
        self.OOD_Datamodule.setup()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.visualise_data()

    def visualise_data(self):
        for i in range(self.Datamodule.n_lines):
            Data_loc = np.where(self.Datamodule.train_labels ==i)[0] # gets all the indices where the label has a certain index (this is correct I believe)
            OOD_Data_loc = np.where(self.OOD_Datamodule.train_labels ==i)[0]
            plt.scatter(self.Datamodule.train_data[Data_loc, 0], self.Datamodule.train_data[Data_loc, 1])
            plt.scatter(self.OOD_Datamodule.train_data[OOD_Data_loc, 0], self.OOD_Datamodule.train_data[OOD_Data_loc, 1])

        plt.savefig('full_visual.png')
        plt.savefig('full_visual.pdf')
        plt.close()
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


#https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib - Used to produce videos from images
class TwoMoonsUncertaintyVisualisation(pl.Callback): # contains methods specifc for the two moons dataset
    def __init__(self,datamodule):
        super().__init__()
        self.datamodule = datamodule

        self.X_vis, self.y_vis = sklearn.datasets.make_moons(n_samples=2500, noise=self.datamodule.noise) # Nawid - moon dataset
        self.X_vis = (self.X_vis - self.datamodule.mean)/self.datamodule.std # normalise data
        self.frames = []
    
    def on_validation_epoch_end(self,trainer, pl_module):
        self.visualise_uncertainty(trainer,pl_module)
    
    def on_test_epoch_end(self,trainer,pl_module):
        self.generate_video()

    def outlier_grid(self): #  Generates the grid of points, outputs, x_lin and y_lin aswell as this is required for the uncertainty visualisation
        domain = 3
        x_lin,y_lin = np.linspace(-domain+0.5, domain+0.5, 50), np.linspace(-domain, domain, 50)

        # Normalising the data which is used for the visualisation
        x_lin, y_lin = (x_lin -self.datamodule.mean[0])/self.datamodule.std[0], (y_lin -self.datamodule.mean[1])/self.datamodule.std[1]
        xx, yy = np.meshgrid(x_lin, y_lin)
        X_grid = np.column_stack([xx.flatten(), yy.flatten()])
        return x_lin, y_lin, X_grid

    def visualise_uncertainty(self,trainer,pl_module):
        # Generates test outlier data
        x_lin, y_lin, X_grid = self.outlier_grid()

        mask = self.y_vis.astype(np.bool)
        centroids = pl_module.update_embeddings(torch.from_numpy(self.X_vis).float().to(pl_module.device),torch.from_numpy(self.y_vis).to(pl_module.device))
        #embeddings = self.model.embedding_encoder.update_embeddings(torch.from_numpy(self.X_vis).float().cuda(),torch.from_numpy(self.y_vis).cuda())
        with torch.no_grad():
            output = pl_module.centroid_confidence(torch.from_numpy(X_grid).float().to(pl_module.device),centroids)
            #output = pl_module(torch.from_numpy(X_grid).float().to(pl_module.device),centroids)
            confidence = output.max(1)[0].cpu().numpy()

        # z = confidence.reshape(xx.shape) # Original version, replaced with x_lin shape[0] since I placed xx in a function
        z = confidence.reshape((x_lin.shape[0], x_lin.shape[0]))
        plt.figure()
        plt.contourf(x_lin, y_lin, z, cmap='cividis')
        plt.colorbar().set_label('Confidence')


        plt.scatter(self.X_vis[mask,0], self.X_vis[mask,1])
        plt.scatter(self.X_vis[~mask,0], self.X_vis[~mask,1])
        #import ipdb; ipdb.set_trace()
        
        video_image = 'Images/file%02d.png' % trainer.current_epoch
        uncertainty_filename = 'Images/uncertainty.png'
        plt.savefig(uncertainty_filename)
        plt.savefig(video_image)
        wandb_uncertainty = 'uncertainty'
        wandb.log({wandb_uncertainty:wandb.Image(uncertainty_filename)})
        # Add the image to the frames section to make a video
        #import ipdb; ipdb.set_trace()
        #self.frames.append(plt.show(block=False))
        plt.close()
    
    def generate_video(self):
        # Changes the directory
        os.chdir("Images")
        video_filename = 'Two_moons_uncertainty.mp4' 
        subprocess.call([
            'ffmpeg', '-framerate', '4', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            video_filename
        ])
        # Logs the video onto wandb
        wandb.log({"Uncertainty visualisation": wandb.Video(video_filename, fps=4, format="mp4")})
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        
        #os.remove(video_filename)

class TwoMoonsRepresentationVisualisation(pl.Callback): # contains methods specifc for the two moons dataset
    def __init__(self,datamodule):
        super().__init__()
        self.datamodule = datamodule

        self.X_vis, self.y_vis = sklearn.datasets.make_moons(n_samples=500, noise=self.datamodule.noise) # Nawid - moon dataset
        self.X_vis = (self.X_vis - self.datamodule.mean)/self.datamodule.std # normalise data
        #self.frames = []
    
    def on_validation_epoch_end(self,trainer, pl_module):
        self.visualise_uncertainty(trainer, pl_module)
    
    def on_test_epoch_end(self,tainer,pl_module):
        self.generate_video()
    
    def visualise_representation(self,trainer,pl_module):
        # Generates test outlier data
        
        mask = self.y_vis.astype(np.bool)
        with torch.no_grad():
            representation = pl_module.callback_vector(torch.from_numpy(self.X_vis).float().to(pl_module.device))

        plt.figure()
        plt.scatter(self.X_vis[mask,0], self.X_vis[mask,1])
        plt.scatter(self.X_vis[~mask,0], self.X_vis[~mask,1])

        plt.scatter(representation[mask,0].cpu(),representation[mask,1].cpu())
        plt.scatter(representation[~mask,0].cpu(),representation[~mask,1].cpu())

        #plt.scatter(self.X_vis[mask,0], self.X_vis[mask,1])
        #plt.scatter(self.X_vis[~mask,0], self.X_vis[~mask,1])
        #import ipdb; ipdb.set_trace()
        
        video_image = 'Images/representation%02d.png' % trainer.current_epoch
        representation_filename = 'Images/representation.png'
        
        plt.savefig(representation_filename)
        plt.savefig(video_image)
        wandb_representation = 'representation'
        wandb.log({wandb_representation:wandb.Image(representation_filename)})
        # Add the image to the frames section to make a video
        #import ipdb; ipdb.set_trace()
        #self.frames.append(plt.show(block=False))
        plt.close()
    
    def generate_video(self):
        # Changes the directory
        os.chdir("Images")
        video_filename = 'Two_moons_representation.mp4' 
        subprocess.call([
            'ffmpeg', '-framerate', '4', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            video_filename
        ])
        # Logs the video onto wandb
        wandb.log({"Representation visualisation": wandb.Video(video_filename, fps=4, format="mp4")})
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        
        #os.remove(video_filename)

