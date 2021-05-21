import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

from Contrastive_uncertainty.general_hierarchy.models.resnet_models import CustomResNet, \
    _custom_resnet, custom_resnet18, custom_resnet34, custom_resnet50