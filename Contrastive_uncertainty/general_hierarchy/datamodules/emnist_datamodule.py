import os
from typing import Optional, Sequence
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split



from warnings import warn
from torchvision import transforms as transform_lib
from torchvision.datasets import EMNIST

from Contrastive_uncertainty.general.datamodules.dataset_normalizations import emnist_normalization
from Contrastive_uncertainty.general.datamodules.datamodule_transforms import dataset_with_indices_emnist
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule