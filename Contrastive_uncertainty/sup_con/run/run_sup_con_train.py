from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.sup_con.models.sup_con_module import SupConModule
from Contrastive_uncertainty.sup_con.models.sup_con_model_instance import ModelInstance

# calls the function
train(sup_con_hparams, SupConModule, ModelInstance)
