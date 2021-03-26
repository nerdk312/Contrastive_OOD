from training.finetune_moco import finetune
from scripts.config.tune_params import finetune_hparams


finetune_hparams['bsz'] =16
finetune_hparams['instance_encoder'] ='resnet18'
finetune_hparams['epochs'] = 1
finetune_hparams['fast_run'] = True
finetune_hparams['training_ratio'] = 0.01
finetune_hparams['validation_ratio'] = 0.2
finetune_hparams['test_ratio'] = 0.2
finetune_hparams['val_check'] = 1

# calls the function
finetune(finetune_hparams)
