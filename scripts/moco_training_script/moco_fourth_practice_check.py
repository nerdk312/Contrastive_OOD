from training.train_moco import train
from scripts.config.default_params import sweep_hparams

sweep_hparams['bsz'] =256
sweep_hparams['epochs'] = 3
sweep_hparams['fast_run'] = False
sweep_hparams['training_ratio'] = 1.0
sweep_hparams['validation_ratio'] = 0.1
sweep_hparams['test_ratio'] = 0.1
sweep_hparams['val_check'] = 1
sweep_hparams['project'] = 'practice' 

train(sweep_hparams)