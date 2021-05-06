import wandb
# Import evaluation methods 
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_trainer_params import trainer_hparams
from Contrastive_uncertainty.cross_entropy.train.resume_cross_entropy import resume

# list of run paths for evaluate
run_paths = ['nerdk312/practice/mrzl2qrz']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    resume(run_path,trainer_hparams)
    