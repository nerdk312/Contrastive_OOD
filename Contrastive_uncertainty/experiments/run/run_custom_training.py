# Import general params
from Contrastive_uncertainty.experiments.config.base_params import base_hparams
from Contrastive_uncertainty.experiments.train.train_custom_experiments import train

train(base_hparams)