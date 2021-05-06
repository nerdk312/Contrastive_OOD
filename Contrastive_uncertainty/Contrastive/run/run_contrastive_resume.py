import wandb
# Import evaluation methods 
from Contrastive_uncertainty.Contrastive.config.contrastive_trainer_params import trainer_hparams
from Contrastive_uncertainty.Contrastive.train.resume_contrastive import resume

# list of run paths for evaluate
run_paths = ['nerdk312/practice/2gxoqmw6']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path,trainer_hparams)
    