from Contrastive_uncertainty.toy_replica.toy_general.train.train_general import train

from Contrastive_uncertainty.toy_replica.vae_models.cross_entropy_vae.config.cross_entropy_vae_params import cross_entropy_vae_hparams
from Contrastive_uncertainty.toy_replica.vae_models.cross_entropy_vae.models.cross_entropy_vae_module import CrossEntropyVAEToy

from Contrastive_uncertainty.toy_replica.vae_models.cross_entropy_vae.models.cross_entropy_vae_model_instance import ModelInstance

# calls the function
train(cross_entropy_vae_hparams, CrossEntropyVAEToy, ModelInstance)