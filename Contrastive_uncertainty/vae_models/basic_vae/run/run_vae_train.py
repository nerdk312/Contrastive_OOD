from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.vae.config.vae_params import vae_hparams
from Contrastive_uncertainty.vae_models.vae.models.vae_module import VAEModule
from Contrastive_uncertainty.vae_models.vae.models.vae_model_instance import ModelInstance



# calls the function
train(vae_hparams, VAEModule, ModelInstance)