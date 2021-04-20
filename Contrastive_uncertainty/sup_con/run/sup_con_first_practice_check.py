from Contrastive_uncertainty.sup_con.train.train_sup_con import train
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams


sup_con_hparams['bsz'] = 16
sup_con_hparams['instance_encoder'] = 'resnet18'
sup_con_hparams['epochs'] = 1
sup_con_hparams['fast_run'] = False
sup_con_hparams['training_ratio'] = 0.01
sup_con_hparams['validation_ratio'] = 0.2
sup_con_hparams['test_ratio'] = 0.2
sup_con_hparams['val_check'] = 1
sup_con_hparams['project'] = 'practice'  # evaluation, contrastive_training
sup_con_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
sup_con_hparams['quick_callback'] = True
train(sup_con_hparams)
