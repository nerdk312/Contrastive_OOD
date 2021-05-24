from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import torch

class Metric():
    def __init__(self, mode, **kwargs):
        self.mode        = mode # Checks the mode, whether inter, intra or intra/inter
        self.requires = ['features', 'target_labels'] # requires labels
        self.name     = 'dists@{}'.format(mode)  # Nawid - based on mode

    def __call__(self, features, target_labels):
        features_locs = [] # Features
        for lab in np.unique(target_labels): # Go through the features
            features_locs.append(np.where(target_labels==lab)[0])

        if 'intra' in self.mode:
            if isinstance(features, torch.Tensor):
                intrafeatures = features.detach().cpu().numpy() # Intra features
            else:
                intrafeatures = features

            intra_dists = []
            for loc in features_locs: #  Goes through the different classes
                c_dists = distance.cdist(intrafeatures[loc], intrafeatures[loc], 'cosine') #  obtain distances between the features belonging to the same class, where the distance is used a cosine I believe
                c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists)) # Obtain sum of differences and then divide
                intra_dists.append(c_dists)
            intra_dists = np.array(intra_dists) # Turn to an array
            maxval      = np.max(intra_dists[1-np.isnan(intra_dists)])
            intra_dists[np.isnan(intra_dists)] = maxval # Change any values of nans into the max val
            intra_dists[np.isinf(intra_dists)] = maxval # Change any values which are inf to the max val
            dist_metric = dist_metric_intra = np.mean(intra_dists) # Mean of intra class distances

        if 'inter' in self.mode: # if inter in the name, calculate inter class distances
            if not isinstance(features, torch.Tensor): # if not torch tensor
                coms = []
                for loc in features_locs:
                    com   = normalize(np.mean(features[loc],axis=0).reshape(1,-1)).reshape(-1) # Normalise the features of a particular class
                    coms.append(com)
                mean_inter_dist = distance.cdist(np.array(coms), np.array(coms), 'cosine') # calculate the distance between the different values using cosine distance
                dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist)) # Calculate the mean inter class distance
            else:
                coms = []
                for loc in features_locs:
                    com   = torch.nn.functional.normalize(torch.mean(features[loc],dim=0).reshape(1,-1), dim=-1).reshape(1,-1) # Normalise
                    coms.append(com)
                mean_inter_dist = 1-torch.cat(coms,dim=0).mm(torch.cat(coms,dim=0).T).detach().cpu().numpy() # Calculate inter class distance
                dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist)) # calculate the sum of inter class distances

        if self.mode=='intra_over_inter': # Calculates intra over inter distance
            dist_metric = dist_metric_intra/np.clip(dist_metric_inter, 1e-8, None)

        return dist_metric
