import wandb
# Import evaluation methods 

from Contrastive_uncertainty.toy_replica.toy_general.train.evaluate_general import evaluation
from Contrastive_uncertainty.toy_replica.sup_con_toy.models.toy_supcon import SupConToy
from Contrastive_uncertainty.toy_replica.sup_con_toy.models.sup_con_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    evaluation(run_path, SupConToy, ModelInstance)

    