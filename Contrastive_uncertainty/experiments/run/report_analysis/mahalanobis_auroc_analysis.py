# Script used to save the mahalanobis AUROC values for the different approaches and datasets
import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math

from ood_centroid_analysis import dataset_dict,key_dict, ood_dataset_string

def AUROC_value(auroc_json):
    auroc_data = np.array(auroc_json['data'])
    auroc = float(auroc_data[-1][1])
    return auroc
    

api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})

# 15 different simulations and each simulation has 2 OOD datasets associated with it
#data_array = np.zeros((30,3)) # potentially could make the shape (3,30)
data_array = np.empty((10,3))
data_array[:] = np.nan


summary_list, config_list, name_list = [], [], []

for i, run in enumerate(runs): 
    #import ipdb; ipdb.set_trace()
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)
    
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    ID_dataset = config_list[i]['dataset']
    model_type = config_list[i]['model_type']


    # .name is the human-readable name of the run.dir
    name_list.append(run.name)
    # Obtain the keys which satisfy the requirement based on https://stackoverflow.com/questions/10484261/find-dictionary-items-whose-key-matches-a-substring/10484313
    keys = [key for key, value in summary_list[i].items() if 'class wise mahalanobis instance fine' in key.lower()]
    # Removes the keys related to the tables images, only retain the Json Files
    auroc_keys = [key for key in keys if 'table' not in key.lower()]
    # Remove keys which use emnist 
    auroc_keys = [key for key in auroc_keys if 'emnist' not in key.lower()]

    # Iterate through the keys (corresponding to different OOD dataset) and save the data
    for key in auroc_keys:
        OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)

        auroc_data_dir = summary_list[i][key]['path']
        with open(auroc_data_dir) as f:
            auroc_json = json.load(f)

        auroc_val = AUROC_value(auroc_json)

        column = key_dict['model_type'][model_type]
        row = 2*key_dict['dataset'][ID_dataset] + dataset_dict[ID_dataset][OOD_dataset]
        data_array[row, column] = auroc_val

# Obtain the names of the rows and the name of the columns
column_names = [model for model in key_dict['model_type'].keys()]

row_names = []
# iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
for dataset in key_dict['dataset'].keys():
    for OOD_dataset in dataset_dict[dataset].keys():
        row_names.append(f'ID:{dataset}, OOD:{OOD_dataset}')


auroc_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
latex_table = auroc_df.to_latex()


# Table post processing
latex_table = latex_table.replace('{}','{Datasets}')
latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
latex_table = latex_table.replace(r"\toprule",r"\hline")
latex_table = latex_table.replace(r"\midrule"," ")
latex_table = latex_table.replace(r"\bottomrule"," ")
#latex_table = latex_table.replace(r"\midrule",r"\hline")
#latex_table = latex_table.replace(r"\bottomrule",r"\hline")
#https://stackoverflow.com/questions/24704299/how-to-treat-t-as-a-regular-string-in-python

latex_table = latex_table.replace(r'\\',r'\\ \hline')
print(latex_table)