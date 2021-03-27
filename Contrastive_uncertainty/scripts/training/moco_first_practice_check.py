from Contrastive_uncertainty.run.train_moco import train
from Contrastive_uncertainty.config.default_params import sweep_hparams


sweep_hparams['bsz'] =16
sweep_hparams['instance_encoder'] ='resnet18'
sweep_hparams['epochs'] = 1
sweep_hparams['fast_run'] = True
sweep_hparams['training_ratio'] = 0.01
sweep_hparams['validation_ratio'] = 0.2
sweep_hparams['test_ratio'] = 0.2
sweep_hparams['val_check'] = 1
sweep_hparams['project'] = 'practice' # evaluation, Moco_training
sweep_hparams['pretrained_network'] = None#'Pretrained_models/finetuned_network.pt'
sweep_hparams['quick_callback'] = True
train(sweep_hparams)
