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
        '''
        self.backbone = nn.Sequential(nn.Linear(2,hidden_dim), nn.ReLU(), 
                                      nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim,emb_dim))
        '''        
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        
        #self.apply(weight_init)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Nawid -output (batch, features)
        return x
    
    '''
    def forward(self, x):
        return torch.nn.functional.normalize(self.backbone(x),dim=1)
    '''