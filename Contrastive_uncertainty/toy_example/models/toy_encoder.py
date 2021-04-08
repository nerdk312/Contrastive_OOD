import torch
import torch.nn.functional as F
from torch import nn

# Neural network
class Backbone(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2,hidden_dim), nn.ReLU(), 
                                      nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim,emb_dim))

    def forward(self, x):
        return torch.nn.functional.normalize(self.backbone(x),dim=1)