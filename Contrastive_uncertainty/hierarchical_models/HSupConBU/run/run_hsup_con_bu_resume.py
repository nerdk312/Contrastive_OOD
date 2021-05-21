import wandb
# Import evaluation methods 


from Contrastive_uncertainty.general_hierarchy.train.resume_general_hierarchy import resume

from Contrastive_uncertainty.hierarchical_models.HSupConBU.config.hsup_con_bu_trainer_params import trainer_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConBU.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_module import HSupConBUModule
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_con_bufig = previous_run.config
    resume(run_path, trainer_hparams, HSupConBUModule, ModelInstance)