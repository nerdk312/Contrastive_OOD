from Contrastive_uncertainty.dpsup_con.train.train_dpsup_con import train
from Contrastive_uncertainty.dpsup_con.config.dpsup_con_params import dpsup_con_hparams


dpsup_con_hparams['bsz'] = 16
dpsup_con_hparams['instance_encoder'] = 'resnet18'
dpsup_con_hparams['epochs'] = 1
dpsup_con_hparams['fast_run'] = False
dpsup_con_hparams['training_ratio'] = 0.01
dpsup_con_hparams['validation_ratio'] = 0.2
dpsup_con_hparams['test_ratio'] = 0.2
dpsup_con_hparams['val_check'] = 1
dpsup_con_hparams['project'] = 'practice'  # evaluation, contrastive_training
dpsup_con_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
dpsup_con_hparams['quick_callback'] = True
train(dpsup_con_hparams)
