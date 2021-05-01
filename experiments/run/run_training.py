# Import general params
from experiments.config.base_params import base_hparams
from experiments.train.train_experiments import train

train(base_hparams)
