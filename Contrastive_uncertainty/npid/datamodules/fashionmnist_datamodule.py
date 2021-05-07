import os
from typing import Optional, Sequence
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split



from warnings import warn
from torchvision import transforms as transform_lib
from torchvision.datasets import FashionMNIST

from Contrastive_uncertainty.npid.datamodules.dataset_normalizations import fashionmnist_normalization
from Contrastive_uncertainty.npid.datamodules.datamodule_transforms import dataset_with_indices

from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule