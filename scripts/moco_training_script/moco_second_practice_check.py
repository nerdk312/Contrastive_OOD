from training.train_moco import train
from scripts.config.default_params import sweep_hparams

sweep_hparams['bsz'] =16
sweep_hparams['instance_encoder'] ='resnet18'
sweep_hparams['epochs'] = 2
sweep_hparams['fast_run'] = False
sweep_hparams['training_ratio'] = 0.01
sweep_hparams['validation_ratio'] = 0.1
sweep_hparams['test_ratio'] = 0.1
sweep_hparams['val_check'] = 1
sweep_hparams['model_saving'] = 1
sweep_hparams['project'] = 'practice' 

train(sweep_hparams)