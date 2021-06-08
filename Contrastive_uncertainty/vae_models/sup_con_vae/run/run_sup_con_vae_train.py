from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.sup_con_vae.config.sup_con_vae_params import sup_con_vae_hparams
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_module import SupConVAEModule
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_model_instance import ModelInstance

# calls the function
train(sup_con_vae_hparams, SupConVAEModule, ModelInstance)