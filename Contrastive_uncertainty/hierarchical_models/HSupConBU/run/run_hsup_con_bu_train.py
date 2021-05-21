from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train
from Contrastive_uncertainty.hierarchical_models.HSupConBU.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_module import HSupConBUModule
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_model_instance import ModelInstance

# calls the function
train(hsup_con_bu_hparams, HSupConBUModule, ModelInstance)
