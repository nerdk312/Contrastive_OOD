from training.train_moco import train
from scripts.config.default_params import sweep_hparams

# calls the function
train(sweep_hparams)