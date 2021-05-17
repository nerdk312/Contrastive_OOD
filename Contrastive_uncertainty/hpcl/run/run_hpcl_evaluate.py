import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general_clustering.train.evaluate_general_clustering import evaluation

from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance


# list of run paths for evaluate
run_paths = ['nerdk312/practice/22zbqp91']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    evaluation(run_path, PCLModule, ModelInstance)

    