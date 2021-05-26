from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.config.hsup_con_bu_centroid_params import hsup_con_bu_centroid_hparams
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroidModule
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance

hsup_con_bu_centroid_hparams['bsz'] = 64
hsup_con_bu_centroid_hparams['instance_encoder'] = 'resnet18'
hsup_con_bu_centroid_hparams['epochs'] = 1
hsup_con_bu_centroid_hparams['fast_run'] = True
hsup_con_bu_centroid_hparams['training_ratio'] = 0.01
hsup_con_bu_centroid_hparams['validation_ratio'] = 0.2
hsup_con_bu_centroid_hparams['test_ratio'] = 0.2
hsup_con_bu_centroid_hparams['val_check'] = 1
hsup_con_bu_centroid_hparams['project'] = 'practice'  # evaluation, contrastive_training
hsup_con_bu_centroid_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
hsup_con_bu_centroid_hparams['quick_callback'] = True
#hsup_con_bu_centroid_hparams['branch_weights'] = [1.0,0,0]
train(hsup_con_bu_centroid_hparams, HSupConBUCentroidModule, ModelInstance)