import wandb
# Import evaluation methods 
from Contrastive_uncertainty.PCL.config.pcl_trainer_params import trainer_hparams
from Contrastive_uncertainty.PCL.train.resume_pcl import resume

# list of run paths for evaluate
run_paths = ['nerdk312/practice/3tb2lutc']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path,trainer_hparams)
    