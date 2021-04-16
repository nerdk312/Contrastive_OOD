from Contrastive_uncertainty.PCL.train.train_pcl import training
from Contrastive_uncertainty.PCL.config.pcl_params import nncl_hparams


nncl_hparams['bsz'] = 128
nncl_hparams['num_cluster'] = [1000]
nncl_hparams['instance_encoder'] = 'resnet18'
nncl_hparams['epochs'] = 1
nncl_hparams['fast_run'] = False
nncl_hparams['training_ratio'] = 0.01
nncl_hparams['validation_ratio'] = 0.2
nncl_hparams['test_ratio'] = 0.2
nncl_hparams['val_check'] = 1
nncl_hparams['project'] = 'practice'  # evaluation, Moco_training
nncl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
nncl_hparams['quick_callback'] = True
training(nncl_hparams)
