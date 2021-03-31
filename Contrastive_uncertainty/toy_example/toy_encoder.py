import torch
import torch.nn.functional as F
from torch import nn

# Neural network
class Backbone(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2,emb_dim), nn.ReLU(), nn.Linear(emb_dim,emb_dim), nn.ReLU(), nn.Linear(emb_dim,2))

    def forward(self, x):
        return torch.nn.functional.normalize(self.backbone(x),dim=1)
