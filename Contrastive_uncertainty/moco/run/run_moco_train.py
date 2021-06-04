from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.moco.models.moco_model_instance import ModelInstance

# calls the function
train(moco_hparams, MocoModule, ModelInstance)
