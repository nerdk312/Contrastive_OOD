# Import general params
from Contrastive_uncertainty.experiments.config.practice_params import practice_hparams
from Contrastive_uncertainty.experiments.train.train_experiments import train

practice_hparams['bsz'] = 256
practice_hparams['num_negatives'] = 1024
practice_hparams['quick_callback'] = False
train(practice_hparams)
