import wandb
# Import evaluation methods 

from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.evaluate_general_hierarchy import evaluation

from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td_centroid.models.hsup_con_td_centroid_module import HSupConTDCentroidToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td_centroid.models.hsup_con_td_centroid_model_instance import ModelInstance

# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_con_bufig = previous_run.config
    evaluation(run_path, HSupConTDCentroidToy, ModelInstance)

    