from Contrastive_uncertainty.npid_pcl.train.train_npid_pcl import train
from Contrastive_uncertainty.npid_pcl.config.npid_pcl_params import npid_pcl_hparams


npid_pcl_hparams['bsz'] = 32
npid_pcl_hparams['instance_encoder'] = 'resnet18'
npid_pcl_hparams['epochs'] = 1
npid_pcl_hparams['fast_run'] = True
npid_pcl_hparams['num_negatives'] = 16
npid_pcl_hparams['training_ratio'] = 0.01
npid_pcl_hparams['validation_ratio'] = 0.2
npid_pcl_hparams['test_ratio'] = 0.2
npid_pcl_hparams['val_check'] = 1
npid_pcl_hparams['project'] = 'practice'  # evaluation, contrastive_training
npid_pcl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
npid_pcl_hparams['quick_callback'] = True
train(npid_pcl_hparams)
