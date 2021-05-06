from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance




pcl_hparams['bsz'] = 128
pcl_hparams['num_cluster'] = [5000]
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
train(pcl_hparams,PCLModule, ModelInstance)
