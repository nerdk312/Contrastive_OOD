import torch
import torch.nn as nn
import torch.nn.functional as F


class FineTuneEncoder(nn.Module):
    def __init__(self,latent_size,feature_map_size= 512,num_classes=10):
        super().__init__()
         # Nawid - preloaded resnet 18 (basic version without changing the kaiming initialisation)
        self.latent_size = latent_size
        self.feature_map_size = feature_map_size
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(self.feature_map_size,self.latent_size)
        self.fc2 = nn.Linear(self.latent_size,self.latent_size)
        # Fully connected layers for the purpose of the classification task
        self.class_fc1 = nn.Linear(self.feature_map_size,self.latent_size)
        self.class_fc2 = nn.Linear(self.latent_size, self.num_classes) # number of classes for the classification problem
        

    # Forward for the instance discrimination task
    def instance_forward(self,x):
        #import ipdb;ipdb.set_trace()
        x = F.relu(self.fc1(x))
        logits = self.fc2(x) # obtain the logits for the class probabilities (logits which is the output before the softmax)

        return logits
    
    def class_forward(self,x):
        x = F.relu(self.class_fc1(x))
        logits = self.class_fc2(x) # obtain the logits for the class probabilities (logits which is the output before the softmax)

        return logits 
    