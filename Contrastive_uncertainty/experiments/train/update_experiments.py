import wandb
from Contrastive_uncertainty.general.train.update_general import update_config

def update(run_paths,update_dict):    
    # Iterate through the run paths
    for run_path in run_paths: 
        # Obtain previous information such as the model type to be able to choose appropriate methods
        update_config(run_path, update_dict)