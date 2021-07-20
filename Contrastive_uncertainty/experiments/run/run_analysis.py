import wandb
import pandas as pd

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams


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

from urllib.request import urlopen
with urlopen("https://wandb.ai/nerdk312/evaluation/runs/1dth3ial/files/media/table/1D%20Background%20Mahalanobis%20emnist_table_1477604_832f9548.table.json") as response:
    source = response.read()
json_file_path = 'https://wandb.ai/nerdk312/evaluation/runs/1dth3ial/files/media/table/1D%20Background%20Mahalanobis%20emnist_table_1477604_832f9548.table.json'
with open(json_file_path, 'r') as j: 
     contents = json.loads(j.read())    


summary_list, config_list, name_list = [], [], []
for run in runs: 
    import ipdb; ipdb.set_trace()
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)
    values = run.summary[]
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

#import ipdb; ipdb.set_trace()

runs_df.to_csv("project.csv")