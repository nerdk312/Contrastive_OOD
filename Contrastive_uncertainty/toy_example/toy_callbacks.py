import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class circular_visualisation(pl.Callback):
    def __init__(self, Datamodule):
        super().__init__()
        self.Datamodule = Datamodule
        #import ipdb; ipdb.set_trace()
        # Obtain class names list for the case of the indomain data

    def on_validation_epoch_end(self,trainer,pl_module):
        collated_features = []
        collated_labels = []
        dataloader = self.Datamodule.train_dataloader()
        for data, labels in dataloader:
            if isinstance(data, tuple) or isinstance(data, list):
                data, *aug_data = data
            data = data.to(pl_module.device)
            feature_vector = pl_module.model.feature_vector(data)
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

        
        





    



    

