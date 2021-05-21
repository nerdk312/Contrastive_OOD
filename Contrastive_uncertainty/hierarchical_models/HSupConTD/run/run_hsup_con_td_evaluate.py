import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general_hierarchy.train.evaluate_general_hierarchy import evaluation


from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_module import HSupConTDModule
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_model_instance import ModelInstance



# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    evaluation(run_path,HSupConTDModule,ModelInstance)

    