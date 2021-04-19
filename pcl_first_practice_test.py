from Contrastive_uncertainty.PCL.train.train_pcl import training
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams


pcl_hparams['bsz'] = 128
pcl_hparams['num_cluster'] = [1000]
pcl_hparams['instance_encoder'] = 'resnet18'
pcl_hparams['epochs'] = 1
pcl_hparams['fast_run'] = False
pcl_hparams['training_ratio'] = 0.01
pcl_hparams['validation_ratio'] = 0.2
pcl_hparams['test_ratio'] = 0.2
pcl_hparams['val_check'] = 1
pcl_hparams['project'] = 'practice'  # evaluation, Moco_training
pcl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
pcl_hparams['quick_callback'] = False #True
training(pcl_hparams)
