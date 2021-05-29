# Import general params
from Contrastive_uncertainty.toy_replica.toy_experiments.config.practice_params import practice_hparams
from Contrastive_uncertainty.toy_replica.toy_experiments.train.train_experiments import train

train(practice_hparams)
