import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m): # Nawid - weight normalisation
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)): # Nawid - if m is an an instance of torch.nn linear or conv2d, then apply weight normalisation
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in",
            nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)




# Neural network
class Backbone(nn.Module):
    def __init__(self,input_dim, hidden_dim, emb_dim):
        super().__init__()
            
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Nawid -output (batch, features)
        return x
