from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import torch

class Metric():
    def __init__(self, t, **kwargs):
        self.t        = t
        self.requires = ['features']
        self.name     = 'uniformity'

    def __call__(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features) # Change to a torch tensor
        uniformity = torch.pdist(features, p=2).pow(2).mul(-self.t).exp().mean().log()
        uniformity = uniformity.item()
        #uniformity = uniformity.detach().cpu().numpy() # Change to numpy to be in the same formate as before
        return uniformity