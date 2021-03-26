import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_models



class MocoEncoder(nn.Module):
    def __init__(self,latent_size,feature_map_size= 512,num_channels= 1,num_classes=10):
        super().__init__()
        self = resnet_models.resnet18(pretrained=False,progress = False,num_classes=latent_size) # Nawid - preloaded resnet 18 (basic version without changing the kaiming initialisation)
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.feature_map_size = feature_map_size
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(
            self.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = nn.Identity()

        # Fully connected layers for the purpose of the classification task
        self.class_fc1 = nn.Linear(self.feature_map_size,self.latent_size)
        self.class_fc2 = nn.Linear(self.latent_size, self.num_classes) # number of classes for the classification problem
        
        '''
        self.resnet = resnet_models.resnet18(pretrained=False,progress = False,num_classes=self.latent_size) # Nawid - preloaded resnet 18 (basic version without changing the kaiming initialisation)
        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            self.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()

        # Fully connected layers for the purpose of the classification task
        self.resnet.class_fc1 = nn.Linear(self.feature_map_size,self.latent_size)
        self.resnet.class_fc2 = nn.Linear(self.latent_size, self.num_classes) # number of classes for the classification problem
        '''

    # Forward for the instance discrimination task
    def forward(self,x):
        x = self(x)
        #x = self.resnet(x)
        return x
    '''
    def class_forward(self,x):
        x = self.resnet.representation_output(x) # Gets the feature map representations which I use for the purpose of pretraining
        x = F.relu(self.resnet.class_fc1(x))
        logits = self.resnet.class_fc2(x) # obtain the logits for the class probabilities (logits which is the output before the softmax)

        return logits 
    '''