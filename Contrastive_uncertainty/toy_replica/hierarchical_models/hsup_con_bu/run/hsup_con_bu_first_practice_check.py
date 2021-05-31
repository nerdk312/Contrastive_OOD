from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_module import HSupConBUToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_model_instance import ModelInstance

hsup_con_bu_hparams['bsz'] = 64
hsup_con_bu_hparams['instance_encoder'] = 'resnet18'
hsup_con_bu_hparams['epochs'] = 1
hsup_con_bu_hparams['fast_run'] = True
hsup_con_bu_hparams['training_ratio'] = 0.01
hsup_con_bu_hparams['validation_ratio'] = 0.2
hsup_con_bu_hparams['test_ratio'] = 0.2
hsup_con_bu_hparams['val_check'] = 1
hsup_con_bu_hparams['project'] = 'practice'  # evaluation, contrastive_training
hsup_con_bu_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hsup_con_bu_hparams['quick_callback'] = True
#import ipdb; ipdb.set_trace()
train(hsup_con_bu_hparams,HSupConBUToy, ModelInstance)
