# Import general params
from Contrastive_uncertainty.experiments.config.base_params import base_hparams
from Contrastive_uncertainty.experiments.train.batch_train_experiments import batch_train

batch_train(base_hparams)