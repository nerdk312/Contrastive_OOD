from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.config.hsup_con_bu_centroid_params import hsup_con_bu_centroid_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroidToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance


# calls the function
train(hsup_con_bu_centroid_hparams, HSupConBUCentroidToy, ModelInstance)
