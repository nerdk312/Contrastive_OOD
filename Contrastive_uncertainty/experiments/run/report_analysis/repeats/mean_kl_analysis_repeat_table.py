# Plotting the centroid distances against the confuson log probability

from numpy.core.fromnumeric import std
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import copy

# Import general params
import json

from Contrastive_uncertainty.experiments.run.report_analysis.ood_centroid_analysis import dataset_dict, ood_dataset_string
from Contrastive_uncertainty.experiments.run.report_analysis.centroid_total_kl_plot_analysis import total_kl_div_vector

# Code for the case of obtaining the MMD distance
def mean_centroid_repeat_table():
    #desired_key = 'Mahalanobis AUROC: instance vector'.lower()
    
    desired_key = 'KL Divergence(Total||Class)'
    decimal_places = 3 # num of decimal places to save

    # Dict for the specific case to the other value
    key_dict = {'model_type':{'Moco':0, 'SupCon':1},
            'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}

    # Dictionary to place values inside
    values_dict = {'Moco':{'MNIST':[],'FashionMNIST':[],'KMNIST':[], 'CIFAR10':[], 'CIFAR100':[]},
    'SupCon':{'MNIST':[],'FashionMNIST':[],'KMNIST':[], 'CIFAR10':[], 'CIFAR100':[]}}
    
    api = wandb.Api()
    data_array = np.empty((5,2))
    data_array[:] = np.nan
    # Array for std values
    std_array = np.empty((5,2))
    std_array[:] = np.nan
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]})
    
    summary_list, config_list, name_list = [], [], []
    root_dir = 'run_data/'
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
        
        # Getting the path
        group_name = config_list[i]['group']
        path_list = runs[i].path
        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
        run_path = '/'.join(path_list)
        run_dir = root_dir + run_path
        data_dir =  summary_list[i][desired_key]['path']
        
        read_dir = run_dir + '/' + data_dir
        name_list.append(run.name)
        #print(run.name)
        #import ipdb; ipdb.set_trace()
        # opening file
        with open(read_dir) as f:
            total_kl_data= json.load(f)
        
        # Calculate the mean distance for a single run and placing into dict
        total_kl_values = total_kl_div_vector(total_kl_data)
        values_dict[model_type][ID_dataset].append(total_kl_values)
    # All values to iterate through
    #Models = ['CE','Moco','SupCon']
    Models = ['Moco','SupCon']
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100']
    

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            column = key_dict['model_type'][Model]
            row = key_dict['dataset'][ID]
            mean_val = np.around(np.mean(values_dict[Model][ID]),decimals=decimal_places) 
            std_val = np.around(np.std(values_dict[Model][ID]),decimals=decimal_places)
            data_array[row, column] = mean_val
            std_array[row, column] = std_val
    
    # iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
    column_names= ['SupCLR' if model=='SupCon' else model for model in key_dict['model_type'].keys()]
    row_names = [dataset for dataset in key_dict['dataset'].keys()]
    
    mean_distance_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
    std_distance_df = pd.DataFrame(std_array, columns = column_names, index = row_names)

    mean_latex_table = mean_distance_df.to_latex()
    std_latex_table =  std_distance_df.to_latex()
    
    mean_values_string = re.findall("\d+\.\d+", mean_latex_table)
    std_values_string = re.findall("\d+\.\d+", std_latex_table)
    
    concat_list_string = [mean_values_string[i] + ' \pm ' + std_values_string[i] for i in range(len(mean_values_string))]
    
    concatenated_list = []
    recursive_string = copy.copy(mean_latex_table)
    for i in range(len(mean_values_string)):
        first_string, recursive_string = recursive_string.split(mean_values_string[i],1) 
        first_string = first_string + mean_values_string[i]
        first_string = first_string.replace(mean_values_string[i], concat_list_string[i])
        concatenated_list.append(first_string)

        #import ipdb ; ipdb.set_trace()

        #mean_latex_table= auroc_mean_latex_table.replace(mean_values_string[i], concat_list_string[i])

    # Concenate the different parts with no space in between 
    mean_latex_table = ''.join(concatenated_list)

    # Table post processing
    mean_latex_table= mean_latex_table.replace('{}','{Datasets}')
    mean_latex_table= mean_latex_table.replace("lrr","|p{3cm}|p{3cm}|p{3cm}|")
    mean_latex_table= mean_latex_table.replace(r"\toprule",r"\hline")
    mean_latex_table = mean_latex_table.replace(r"\\",r"\\ \hline")
    mean_latex_table= mean_latex_table.replace(r"\midrule"," ")
    mean_latex_table= mean_latex_table.replace(r"\bottomrule"," ")

    mean_latex_table = mean_latex_table + ' \\\ \hline'

    print(mean_latex_table)

if __name__ == '__main__':
    mean_centroid_repeat_table()
