import torch
import torch.nn.functional as F
from torch import nn


def weight_init(m): # Nawid - weight normalisation
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)): # Nawid - if m is an an instance of torch.nn linear or conv2d, then apply weight normalisation
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in",
            nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)

# Neural network
class Backbone(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super().__init__()
               
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        
        self.apply(weight_init)
    # Code required for memory bank version    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Nawid -output (batch, features)
        #x = torch.nn.functional.normalize(x,dim = 1)
        return x


    # Code required for the second case
    def pairwise_l2_sq(self,x): #Maths explained in https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        """Compute pairwise squared Euclidean distances.
        """

        dot = torch.mm(x, torch.t(x))
        norm_sq = torch.diag(dot)
        dist = norm_sq[None, :] - 2*dot + norm_sq[:, None]
        dist = torch.clamp(dist, min=0)  # replace negative values with 0 which could potentially arise from numerical imprecision with zeros
        return dist.float() #shape: (batch,batch)
        
    # Original distances method
    def original_dists(self, x): #  input of y is not one hot encoding, specific values for each class
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)  # Nawid -output (batch, features)
        distances = self.pairwise_l2_sq(z)
        return distances

    @staticmethod
    def _softmax(x):
        """Compute row-wise softmax.
        Notes:
        Since the input to this softmax is the negative of the
        pairwise L2 distances, we don't need to do the classical
        numerical stability trick.
        """
        z = x - torch.max(x, dim=1, keepdim=True)[0] # ADDED THIS, required for numerical stability
        # Add a tiny constant for stability of log we take later
        #z = z + 1e-8
        exp = torch.exp(z)
        return exp / (exp.sum(dim=1) - torch.diagonal(exp, 0))

    
    
    