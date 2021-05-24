from Contrastive_uncertainty.toy_replica.toy_general.train.train_general import train
from Contrastive_uncertainty.toy_replica.sup_con_toy.config.sup_con_params import sup_con_hparams


from Contrastive_uncertainty.toy_replica.sup_con_toy.models.toy_supcon import SupConToy
from Contrastive_uncertainty.toy_replica.sup_con_toy.models.sup_con_model_instance import ModelInstance

# calls the function
train(sup_con_hparams, SupConToy, ModelInstance)
