import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from Contrastive_uncertainty.general.models.resnet_models import BasicBlock, Bottleneck, ResNet,\
    _resnet, resnet18, resnet34, resnet50