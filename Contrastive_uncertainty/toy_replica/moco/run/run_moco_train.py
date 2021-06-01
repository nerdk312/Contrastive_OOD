from Contrastive_uncertainty.toy_replica.toy_general.train.train_general import train
from Contrastive_uncertainty.toy_replica.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.toy_replica.moco.models.moco_module import MocoToy
from Contrastive_uncertainty.toy_replica.moco.models.moco_model_instance import ModelInstance


# calls the function
train(moco_hparams,MocoToy, ModelInstance)
