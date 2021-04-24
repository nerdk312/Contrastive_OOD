from Contrastive_uncertainty.unsup_con.train.train_unsup_con import train
from Contrastive_uncertainty.unsup_con.config.unsup_con_params import unsup_con_hparams


unsup_con_hparams['bsz'] = 16
unsup_con_hparams['instance_encoder'] = 'resnet18'
unsup_con_hparams['epochs'] = 1
unsup_con_hparams['fast_run'] = False
unsup_con_hparams['training_ratio'] = 0.01
unsup_con_hparams['validation_ratio'] = 0.2
unsup_con_hparams['test_ratio'] = 0.2
unsup_con_hparams['val_check'] = 1
unsup_con_hparams['project'] = 'practice'  # evaluation, contrastive_training
unsup_con_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
unsup_con_hparams['quick_callback'] = True
train(unsup_con_hparams)
