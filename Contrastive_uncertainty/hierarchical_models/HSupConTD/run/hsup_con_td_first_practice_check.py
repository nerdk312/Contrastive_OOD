from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.hierarchical_models.HSupConTD.config.hsup_con_td_params import hsup_con_td_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_module import HSupConTDModule
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_model_instance import ModelInstance


hsup_con_td_hparams['bsz'] = 64
hsup_con_td_hparams['instance_encoder'] = 'resnet18'
hsup_con_td_hparams['epochs'] = 1
hsup_con_td_hparams['fast_run'] = True
hsup_con_td_hparams['training_ratio'] = 0.01
hsup_con_td_hparams['validation_ratio'] = 0.2
hsup_con_td_hparams['test_ratio'] = 0.2
hsup_con_td_hparams['val_check'] = 1
hsup_con_td_hparams['project'] = 'practice'  # evaluation, contrastive_training
hsup_con_td_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hsup_con_td_hparams['quick_callback'] = True

train(hsup_con_td_hparams,HSupConTDModule, ModelInstance)
