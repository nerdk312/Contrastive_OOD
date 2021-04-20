from Contrastive_uncertainty.cross_entropy.train.train_cross_entropy import train
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params  import cross_entropy_hparams


cross_entropy_hparams['bsz'] = 16
cross_entropy_hparams['instance_encoder'] = 'resnet18'
cross_entropy_hparams['epochs'] = 1
cross_entropy_hparams['fast_run'] = False
cross_entropy_hparams['training_ratio'] = 0.01
cross_entropy_hparams['validation_ratio'] = 0.2
cross_entropy_hparams['test_ratio'] = 0.2
cross_entropy_hparams['val_check'] = 1
cross_entropy_hparams['project'] = 'practice'  # evaluation, Moco_training
cross_entropy_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
cross_entropy_hparams['quick_callback'] = True
train(cross_entropy_hparams)
