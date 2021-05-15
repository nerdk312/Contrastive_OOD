import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general_clustering.train.resume_general_clustering import resume

from Contrastive_uncertainty.multi_PCL.config.multi_pcl_trainer_params import trainer_hparams
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance


# list of run paths for evaluate
run_paths = ['nerdk312/practice/3qjd0ksl']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path, trainer_hparams, MultiPCLModule, ModelInstance)
