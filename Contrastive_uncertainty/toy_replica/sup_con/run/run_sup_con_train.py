from Contrastive_uncertainty.toy_replica.toy_general.train.train_general import train
from Contrastive_uncertainty.toy_replica.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.toy_replica.sup_con.models.sup_con_module import SupConToy 
from Contrastive_uncertainty.toy_replica.sup_con.models.sup_con_model_instance import ModelInstance


# calls the function
train(sup_con_hparams, SupConToy, ModelInstance)
