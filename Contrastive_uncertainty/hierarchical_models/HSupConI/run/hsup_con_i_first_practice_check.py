from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.hierarchical_models.HSupConI.config.hsup_con_i_params import hsup_con_i_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConI.models.hsup_con_i_module import HSupConIModule
from Contrastive_uncertainty.hierarchical_models.HSupConI.models.hsup_con_i_model_instance import ModelInstance


hsup_con_i_hparams['bsz'] = 16
hsup_con_i_hparams['instance_encoder'] = 'resnet18'
hsup_con_i_hparams['epochs'] = 1
hsup_con_i_hparams['fast_run'] = True
hsup_con_i_hparams['training_ratio'] = 0.01
hsup_con_i_hparams['validation_ratio'] = 0.2
hsup_con_i_hparams['test_ratio'] = 0.2
hsup_con_i_hparams['val_check'] = 1
hsup_con_i_hparams['project'] = 'practice'  # evaluation, contrastive_training
hsup_con_i_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hsup_con_i_hparams['quick_callback'] = True

train(hsup_con_i_hparams,HSupConIModule, ModelInstance)
