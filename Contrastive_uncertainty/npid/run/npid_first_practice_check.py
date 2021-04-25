from Contrastive_uncertainty.npid.train.train_npid import train
from Contrastive_uncertainty.npid.config.npid_params import npid_hparams


npid_hparams['bsz'] = 16
npid_hparams['instance_encoder'] = 'resnet18'
npid_hparams['epochs'] = 1
npid_hparams['fast_run'] = False
npid_hparams['training_ratio'] = 0.01
npid_hparams['validation_ratio'] = 0.2
npid_hparams['test_ratio'] = 0.2
npid_hparams['val_check'] = 1
npid_hparams['project'] = 'practice'  # evaluation, contrastive_training
npid_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
npid_hparams['quick_callback'] = True
train(npid_hparams)
