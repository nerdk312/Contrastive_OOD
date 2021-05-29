# Import general params
from Contrastive_uncertainty.toy_replica.toy_experiments.config.base_params import base_hparams
from Contrastive_uncertainty.toy_replica.toy_experiments.train.train_experiments import train

train(base_hparams)