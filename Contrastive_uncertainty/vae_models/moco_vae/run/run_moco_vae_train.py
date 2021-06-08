from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.moco_vae.config.moco_vae_params import moco_vae_hparams
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_module import MocoVAEModule
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_model_instance import ModelInstance



# calls the function
train(moco_vae_hparams, MocoVAEModule, ModelInstance)
