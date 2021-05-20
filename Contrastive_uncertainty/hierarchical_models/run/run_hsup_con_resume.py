import wandb
# Import evaluation methods 


from Contrastive_uncertainty.general_hierarchy.train.resume_general_hierarchy import resume

from Contrastive_uncertainty.hierarchical_models.config.hsup_con_trainer_params import trainer_hparams
from Contrastive_uncertainty.hierarchical_models.config.hsup_con_params import hsup_con_hparams
from Contrastive_uncertainty.hierarchical_models.models.hsup_con_module import HSupConModule
from Contrastive_uncertainty.hierarchical_models.models.hsup_con_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path, trainer_hparams, HSupConModule, ModelInstance)