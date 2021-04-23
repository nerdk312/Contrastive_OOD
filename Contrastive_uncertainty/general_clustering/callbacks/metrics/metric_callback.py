import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from PIL import Image
import faiss


import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import copy

from Contrastive_uncertainty.general.general_pl_callbacks.metrics.metric_callback import MetricLogger,evaluation_metrics, evaltypes