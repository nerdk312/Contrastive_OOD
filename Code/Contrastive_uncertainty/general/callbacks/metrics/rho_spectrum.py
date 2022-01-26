
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import wandb



class Metric():
    def __init__(self, embed_dim, mode,  **kwargs):
        self.mode      = mode
        self.embed_dim = embed_dim
        self.requires = ['features']
        self.name     = 'rho_spectrum@'+str(mode)

    def __call__(self, features):
        from sklearn.decomposition import TruncatedSVD
        from scipy.stats import entropy
        import torch

        if isinstance(features, torch.Tensor):
            _,s,_ = torch.svd(features)  # Nawid - obtains singular values of svd
            s     = s.cpu().numpy()
        else:
            svd = TruncatedSVD(n_components=self.embed_dim-1, n_iter=7, random_state=42)
            svd.fit(features)
            s = svd.singular_values_
        
        if self.mode!=0:
            s = s[np.abs(self.mode)-1:] # Nawid - select a subset of the values (from the mode -1 value to the last value)
        s_norm  = s/np.sum(s) # Nawid- normalise by sum of spectral values
        
        uniform = np.ones(len(s))/(len(s))

        if self.mode<0:
            kl = entropy(s_norm, uniform) # Nawid - calculate the KL divergence between a uniform distribution and the singular values 
        if self.mode>0:
            kl = entropy(uniform, s_norm)
        if self.mode==0:
            kl = s_norm
            '''
            # Log the value for the first 3 singular values
            wandb.log({f'singular value {0}':s_norm[0]})
            wandb.log({f'singular value {1}':s_norm[1]})
            wandb.log({f'singular value {2}':s_norm[2]})
            wandb.log({f'singular value {10}':s_norm[10]})
            '''
        return kl
