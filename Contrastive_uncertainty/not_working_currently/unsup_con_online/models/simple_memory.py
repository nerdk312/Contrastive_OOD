import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import get_dist_info
from Contrastive_uncertainty.unsup_con_online.models.alias_multinomial import AliasMethod

class SimpleMemory(nn.Module):
    """Simple memory bank for NPID.
    Args:
        length (int): Number of features stored in the memory bank.
        feat_dim (int): Dimension of stored features.
        momentum (float): Momentum coefficient for updating features.
    """

    def __init__(self, length, feat_dim, momentum, **kwargs):
        super(SimpleMemory, self).__init__()
        # Information on the distribution of the data
        self.rank, self.num_replicas = get_dist_info()
        # initialise feature bank and then normalise
        self.feature_bank = torch.randn(length, feat_dim).cuda()
        self.feature_bank = nn.functional.normalize(self.feature_bank)
        # Momentum for feature bank
        self.momentum = momentum
        # Used to sample from feature bank
        self.multinomial = AliasMethod(torch.ones(length))
        self.multinomial.cuda()

    def update(self, ind, feature):
        """Update features in memory bank.
        Args:
            ind (Tensor): Indices for the batch of features.
            feature (Tensor): Batch of features.
        """
        # Normalise the feature
        feature_norm = nn.functional.normalize(feature)
        # Obtain indices and features from the different gpus
        ind, feature_norm = self._gather(ind, feature_norm)
        # Obtain the old features from the feature bank
        feature_old = self.feature_bank[ind, ...]
        # Obtain new features from running average of data
        feature_new = (1 - self.momentum) * feature_old + \
            self.momentum * feature_norm
        # Normalise new features
        feature_new_norm = nn.functional.normalize(feature_new)
        # Update featrue bank
        self.feature_bank[ind, ...] = feature_new_norm
    # Nawid - gather information such as the features and the indices from multiple gpus
    def _gather(self, ind, feature):
        """Gather indices and features.
        Args:
            ind (Tensor): Indices for the batch of features.
            feature (Tensor): Batch of features.
        Returns:
            Tensor: Gathered indices.
            Tensor: Gathered features.
        """
        ind_gathered = [
            torch.ones_like(ind).cuda() for _ in range(self.num_replicas)
        ]
        feature_gathered = [
            torch.ones_like(feature).cuda() for _ in range(self.num_replicas)
        ]
        dist.all_gather(ind_gathered, ind)
        dist.all_gather(feature_gathered, feature)
        ind_gathered = torch.cat(ind_gathered, dim=0)
        feature_gathered = torch.cat(feature_gathered, dim=0)
        return ind_gathered, feature_gathered