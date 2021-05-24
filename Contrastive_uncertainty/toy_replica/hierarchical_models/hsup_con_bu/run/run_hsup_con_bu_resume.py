import wandb
# Import evaluation methods 

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.resume_general_hierarchy import resume

from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_trainer_params import trainer_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_module import HSupConBUToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_con_bufig = previous_run.config
    resume(run_path, trainer_hparams, HSupConBUToy, ModelInstance)