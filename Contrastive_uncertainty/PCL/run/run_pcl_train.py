from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance


# calls the function
train(pcl_hparams,PCLModule, ModelInstance)