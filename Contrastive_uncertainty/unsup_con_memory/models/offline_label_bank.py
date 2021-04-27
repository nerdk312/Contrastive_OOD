import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.distributed as dist

from Contrastive_uncertainty.general_clustering.models.offline_label_bank import OfflineLabelMemory
from Contrastive_uncertainty.unsup_con_memory.utils.alias_multinomial import AliasMethod
    