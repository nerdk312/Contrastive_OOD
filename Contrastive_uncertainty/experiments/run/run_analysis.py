import wandb
import pandas as pd

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams
import json


# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})

'''
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)
'''
'''
from urllib.request import urlopen
with urlopen("https://wandb.ai/nerdk312/evaluation/runs/1dth3ial/files/media/table/1D%20Background%20Mahalanobis%20emnist_table_1477604_832f9548.table.json") as response:
    source = response.read()
json_file_path = 'https://wandb.ai/nerdk312/evaluation/runs/1dth3ial/files/media/table/1D%20Background%20Mahalanobis%20emnist_table_1477604_832f9548.table.json'
with open(json_file_path, 'r') as j: 
     contents = json.loads(j.read())  
'''

summary_list, config_list, name_list = [], [], []
# Goes from the oldest to the newest 
#for run in reversed(runs): 
for run in runs: 
    #import ipdb; ipdb.set_trace()
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    values = run.summary
    summary_list.append(run.summary._json_dict)
    
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.dir
    name_list.append(run.name)
    
    #import ipdb; ipdb.set_trace()

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

#import ipdb; ipdb.set_trace()

runs_df.to_csv("project.csv")



# Approach 1 of loading file (download file and then open)
json.load(run.file("wandb-metadata.json").download())


# Approach 2 - Allows opening the json file from the filename after the file has been saved
with open('wandb-metadata.json') as f: 
    d = json.load(f) 
    print(d) 

# Obtain the dataset for the task
config_list[0]['dataset'] 


data_dir = summary_list[0]['Centroid Distances Average vector_table']['path']                                                                                                                                                                                    
file_data = json.load(run.file(data_dir).download()) 

data = file_data['data']