import wandb
# Import evaluation methods 


from Contrastive_uncertainty.general_hierarchy.train.resume_general_hierarchy import resume

from Contrastive_uncertainty.hierarchical_models.HSupConI.config.hsup_con_i_trainer_params import trainer_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConI.config.hsup_con_i_params import hsup_con_i_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConI.models.hsup_con_i_module import HSupConIModule
from Contrastive_uncertainty.hierarchical_models.HSupConI.models.hsup_con_i_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_con_ifig = previous_run.config
    resume(run_path, trainer_hparams, HSupConIModule, ModelInstance)