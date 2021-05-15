from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance

# calls the function
train(unsup_con_memory_hparams,UnSupConMemoryModule, ModelInstance)