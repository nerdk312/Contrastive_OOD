# Script for saving the relevant files onto my local machine



import wandb
import pandas as pd
import numpy as np

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams
import json


api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})

summary_list, config_list, name_list = [], [], []

bsz = 10
# Where to save data for the different runs
root_dir = 'run_data/'
for i, run in enumerate(runs): 
    # .summary contains the output keys/values for metrics like accuracy.
    summary_list.append(run.summary._json_dict)
    # Get the run path of the data
    run_path = '/'.join(runs[i].path)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})
    
    # .name is the human-readable name of the run.dir
    name_list.append(run.name)
    # Obtain the keys which satisfy the requirement based on https://stackoverflow.com/questions/10484261/find-dictionary-items-whose-key-matches-a-substring/10484313
    
    keys = [key for key, value in summary_list[i].items() if 'typicality one vs ood rest' in key.lower()]
    
    # Removes the keys related to the tables images, only retain the Json Files
    keys = [key for key in keys if f'batch size {bsz}' in key.lower()]
        
    for key in keys:
        
        data_dir = summary_list[i][key]['path'] 
        run_dir = root_dir + run_path

        '''
        read_dir = run_dir + '/' + data_dir
        with open(read_dir) as f: 
            data = json.load(f) 
        '''

        file_data = json.load(run.file(data_dir).download(root=run_dir)) 
    
    
    # Obtain the dataset and the model type

    
    
    '''
    with open(data_dir) as f: 
        data = json.load(f) 
    '''