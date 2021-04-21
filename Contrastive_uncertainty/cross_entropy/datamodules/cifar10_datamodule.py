import os
from typing import Optional, Sequence
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from Contrastive_uncertainty.cross_entropy.datamodules.dataset_normalizations import cifar10_normalization
from Contrastive_uncertainty.cross_entropy.datamodules.datamodule_transforms import dataset_with_indices
from warnings import warn

from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10

from Contrastive_uncertainty.general.datamodules.cifar10_datamodule import CIFAR10DataModule


