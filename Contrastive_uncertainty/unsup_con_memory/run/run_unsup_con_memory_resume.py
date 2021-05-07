import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general_clustering.train.resume_general_clustering import resume

from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_trainer_params import trainer_hparams
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance


# list of run paths for evaluate
run_paths = ['nerdk312/practice/tb5gmc1c']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path, trainer_hparams, UnSupConMemoryModule, ModelInstance)