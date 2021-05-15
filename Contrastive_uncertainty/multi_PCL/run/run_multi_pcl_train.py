from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.multi_PCL.config.multi_pcl_params  import multi_pcl_hparams
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance


# calls the function
train(multi_pcl_hparams,MultiPCLModule, ModelInstance)