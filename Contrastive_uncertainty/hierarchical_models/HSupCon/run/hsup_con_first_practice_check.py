from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.hierarchical_models.HSupCon.config.hsup_con_params import hsup_con_hparams
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_module import HSupConModule
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_model_instance import ModelInstance


hsup_con_hparams['bsz'] = 16
hsup_con_hparams['instance_encoder'] = 'resnet18'
hsup_con_hparams['epochs'] = 1
hsup_con_hparams['fast_run'] = True
hsup_con_hparams['training_ratio'] = 0.01
hsup_con_hparams['validation_ratio'] = 0.2
hsup_con_hparams['test_ratio'] = 0.2
hsup_con_hparams['val_check'] = 1
hsup_con_hparams['project'] = 'practice'  # evaluation, contrastive_training
hsup_con_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hsup_con_hparams['quick_callback'] = True

train(hsup_con_hparams,HSupConModule, ModelInstance)
