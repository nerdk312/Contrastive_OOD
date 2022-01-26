import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.distributed as dist

from Contrastive_uncertainty.general.utils.alias_multinomial import AliasMethod

class OfflineLabelMemory(nn.Module):
    """Memory modules for Label case.
    Args:
        length (int): Number of features stored in samples memory.
        feat_dim (int): Dimension of stored features.
        memory_momentum (float): Momentum coefficient for updating features.
        num_classes (int): Number of clusters.
        
    """

    def __init__(self, length, feat_dim, memory_momentum, num_classes,
                 **kwargs):
        super(OfflineLabelMemory, self).__init__()
        #self.rank, self.num_replicas = get_dist_info()
        #if self.rank == 0:
        # Make feature bank
        self.feature_bank = torch.zeros((length, feat_dim),
                                        dtype=torch.float32, device='cuda')
        # Make label bank
        self.label_bank = torch.zeros((length, ), dtype=torch.long, device='cuda')
        
        
        self.feat_dim = feat_dim
        self.initialized = False
        self.memory_momentum = memory_momentum
        self.num_classes = num_classes
        self.multinomial = AliasMethod(torch.ones(length))
        self.multinomial.cuda()

    def init_memory(self,model,feature, label):
        """Initialize memory modules."""
        #import ipdb; ipdb.set_trace()  
        self.initialized = True
        # Copy labels 
        self.label_bank.copy_(torch.from_numpy(label).long().to(model.device))
        # make sure no empty clusters
        assert (np.bincount(label, minlength=self.num_classes) != 0).all()
        # Copy features of the data 
        feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10)
        self.feature_bank.copy_(torch.from_numpy(feature).to(model.device))
            

    
    def update_samples_memory(self, ind, feature, label):
        """Update samples memory."""

        assert self.initialized
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        
        # change indices to cpu
        #ind = ind.cpu()
        #if self.rank == 0:
        # Select the old features
        feature_old = self.feature_bank[ind, ...]#.cuda()
        # Update the old features
        feature_new = self.memory_momentum * feature_old + \
            (1 - self.memory_momentum) * feature_norm
        feature_norm = feature_new / (
            feature_new.norm(dim=1).view(-1, 1) + 1e-10)
        # Update feature bank
        self.feature_bank[ind, ...] = feature_norm #.cpu()
        
        # update feature bank  
        newlabel = label #.cpu()
        self.label_bank[ind] = newlabel.clone()  # copy to cpu
        
    

    