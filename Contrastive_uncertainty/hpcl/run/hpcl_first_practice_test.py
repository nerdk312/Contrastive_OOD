from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.hpcl.config.hpcl_params import hpcl_hparams
from Contrastive_uncertainty.hpcl.models.hpcl_module import HPCLModule
from Contrastive_uncertainty.hpcl.models.hpcl_model_instance import ModelInstance


hpcl_hparams['bsz'] = 16
hpcl_hparams['instance_encoder'] = 'resnet18'
hpcl_hparams['epochs'] = 1
hpcl_hparams['fast_run'] = True
hpcl_hparams['training_ratio'] = 0.01
hpcl_hparams['validation_ratio'] = 0.2
hpcl_hparams['test_ratio'] = 0.2
hpcl_hparams['val_check'] = 1
hpcl_hparams['project'] = 'practice'  # evaluation, Moco_training
hpcl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hpcl_hparams['quick_callback'] = True  # True
train(hpcl_hparams, HPCLModule, ModelInstance)
