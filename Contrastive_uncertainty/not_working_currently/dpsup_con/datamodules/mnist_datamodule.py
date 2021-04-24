import os
from typing import Optional, Sequence
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from warnings import warn
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST

from Contrastive_uncertainty.dpsup_con.datamodules.dataset_normalizations import mnist_normalization
from Contrastive_uncertainty.dpsup_con.datamodules.datamodule_transforms import dataset_with_indices

from Contrastive_uncertainty.general_clustering.datamodules.mnist_datamodule import MNISTDataModule