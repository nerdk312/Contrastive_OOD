from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_module import HSupConBUToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_model_instance import ModelInstance


# calls the function
train(hsup_con_bu_hparams, HSupConBUToy, ModelInstance)