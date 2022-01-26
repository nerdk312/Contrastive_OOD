import wandb
# Import evaluation methods 

from Contrastive_uncertainty.general.train.evaluate_general import evaluation

from Contrastive_uncertainty.Contrastive.models.contrastive_module import ContrastiveModule
from Contrastive_uncertainty.Contrastive.models.contrastive_model_instance import ModelInstance


# list of run paths for evaluate
run_paths = ['nerdk312/practice/76dumko3']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    evaluation(run_path,ContrastiveModule,ModelInstance)

    