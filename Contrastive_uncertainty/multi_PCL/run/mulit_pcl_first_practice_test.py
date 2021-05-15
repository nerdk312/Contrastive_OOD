from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.multi_PCL.config.multi_pcl_params import multi_pcl_hparams
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance


pcl_hparams['bsz'] = 128
pcl_hparams['num_cluster'] = [5000]
pcl_hparams['instance_encoder'] = 'resnet18'
pcl_hparams['epochs'] = 1
pcl_hparams['fast_run'] = True
pcl_hparams['training_ratio'] = 0.01
pcl_hparams['validation_ratio'] = 0.2
pcl_hparams['test_ratio'] = 0.2
pcl_hparams['val_check'] = 1
pcl_hparams['project'] = 'practice'  # evaluation, Moco_training
pcl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
pcl_hparams['quick_callback'] = True #True
train(multi_pcl_hparams,MultiPCLModule, ModelInstance)
