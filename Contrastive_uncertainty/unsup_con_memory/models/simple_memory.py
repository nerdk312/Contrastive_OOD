import torch
import torch.nn as nn
import torch.distributed as dist
#from mmcv.runner import get_dist_info

from Contrastive_uncertainty.unsup_con_memory.utils.alias_multinomial import AliasMethod
from Contrastive_uncertainty.general_clustering.models.simple_memory import SimpleMemory