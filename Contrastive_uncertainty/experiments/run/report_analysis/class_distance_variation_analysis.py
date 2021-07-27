# Script used to place the intra and inter class distances in a table

import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math

from ood_centroid_analysis import key_dict


api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})

#parameter = 'dists@intra: instance: fine'  
parameter = 'dists@inter: instance: fine'
#parameter = 'dists@intra_over_inter: instance: fine'
# Five different datasets with 3 models
data_array = np.empty((5,3))
data_array[:] = np.nan
summary_list, config_list = [], []

for i, run in enumerate(runs): 
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

    ID_dataset = config_list[i]['dataset']
    model_type = config_list[i]['model_type']
    
    value = summary_list[i][parameter]
    column = key_dict['model_type'][model_type]
    row = key_dict['dataset'][ID_dataset]
    data_array[row, column] = np.around(value,decimals=3)

# Obtain the names of the rows and the name of the columns
column_names = [model for model in key_dict['model_type'].keys()]
row_names = [dataset for dataset in key_dict['dataset'].keys()]
data_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
latex_table = data_df.to_latex()
print(latex_table)