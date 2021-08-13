# Script used to save the mahalanobis AUROC values for the different approaches and datasets
import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math


import re
from ood_centroid_analysis import dataset_dict,key_dict
    


api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})

# 15 different simulations and each simulation has 2 OOD datasets associated with it
#data_array = np.zeros((30,3)) # potentially could make the shape (3,30)
max_data_array = np.empty((10,1))
max_data_array[:] = np.nan
max_desired_key = 'Max Class Wise CLP'


min_data_array = np.empty((10,1))
min_data_array[:] = np.nan
min_desired_key = 'Min Class Wise CLP'
decimal_places = 2

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
    OOD_dataset = config_list[i]['OOD_dataset'][0]
    
    row = 2*key_dict['dataset'][ID_dataset] + dataset_dict[ID_dataset][OOD_dataset]
    column = 0

    max_value = summary_list[i][max_desired_key]
    max_data_array[row, column] = round(max_value,2)

    min_value = summary_list[i][min_desired_key]
    min_data_array[row, column] = round(min_value,2)

column_names = ['CLP']
row_names = []
# iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
for dataset in key_dict['dataset'].keys():
    for OOD_dataset in dataset_dict[dataset].keys():
        row_names.append(f'ID: {dataset}, OOD: {OOD_dataset}')


min_values = pd.DataFrame(min_data_array, columns = column_names, index = row_names)
min_latex_table = min_values.to_latex()
print(min_latex_table)

max_values = pd.DataFrame(max_data_array, columns = column_names, index = row_names)
max_latex_table = max_values.to_latex()
print(max_latex_table)



# Replacing the values with a concatenated list
#https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
min_values_string = re.findall("\d+\.\d+", min_latex_table)
max_values_string = re.findall("\d+\.\d+", max_latex_table)
concat_list_string = [min_values_string[i] + ' to ' + max_values_string[i] for i in range(len(min_values_string))]

# https://www.programiz.com/python-programming/methods/string/replace#:~:text=from%20replace()-,The%20replace()%20method%20returns%20a%20copy%20of%20the%20string,copy%20of%20the%20original%20string.
for i in range(len(min_values_string)):
    min_latex_table = min_latex_table.replace(min_values_string[i], concat_list_string[i])


# Table post processing
# Replace the {} with {Datasets}
min_latex_table = min_latex_table.replace('{}','{Datasets}')

min_latex_table = min_latex_table.replace("lr","|l|c|")
min_latex_table = min_latex_table.replace(r"\toprule",r"\hline")
min_latex_table = min_latex_table.replace(r"\midrule"," ")
min_latex_table = min_latex_table.replace(r"\bottomrule"," ")

#min_latex_table = min_latex_table.replace(r"\midrule",r"\hline")
#min_latex_table = min_latex_table.replace(r"\bottomrule",r"\hline")
#https://stackoverflow.com/questions/24704299/how-to-treat-t-as-a-regular-string-in-python

min_latex_table = min_latex_table.replace(r'\\',r'\\ \hline')

print(min_latex_table)


#import ipdb; ipdb.set_trace()
#limit_string = [' '.join([min_values_string[i],'to', min_values_string[i]]) for i in range(len(min_values_string))]

#import ipdb; ipdb.set_trace()
#import ipdb; ipdb.set_trace()
'''

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
        row_names.append(f'ID: {dataset}, OOD: {OOD_dataset}')


auroc_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
latex_table = auroc_df.to_latex()
print(latex_table)
'''