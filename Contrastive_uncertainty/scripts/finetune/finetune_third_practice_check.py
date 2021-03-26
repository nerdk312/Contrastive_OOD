from run.finetune_moco import finetune
from Contrastive_uncertainty.config.tune_params import finetune_hparams


finetune_hparams['bsz'] =16
finetune_hparams['epochs'] = 2
finetune_hparams['fast_run'] = False
finetune_hparams['training_ratio'] = 0.1
finetune_hparams['validation_ratio'] = 0.1
finetune_hparams['test_ratio'] = 0.1
finetune_hparams['val_check'] = 1

# calls the function
finetune(finetune_hparams)
