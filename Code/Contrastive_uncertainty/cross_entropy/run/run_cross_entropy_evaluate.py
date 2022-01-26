import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general.train.evaluate_general import evaluation

from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance


# list of run paths for evaluate
run_paths = ['nerdk312/practice/123f07wn']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    evaluation(run_path,CrossEntropyModule,ModelInstance)

    