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

from Contrastive_uncertainty.experiments.run.report_analysis.ood_centroid_analysis import dataset_dict,key_dict, ood_dataset_string

# Code for the case of obtaining the point and marginal typicality OOD values for the different repeats of the data
def ood_detection_repeat_table():
    #desired_key = 'Mahalanobis AUROC: instance vector'.lower()
    
    desired_key = 'Marginal Typicality Entropy Average Likelihood Batch Size'.lower()
    decimal_places = 3 # num of decimal places to save


    # Dictionary to place values inside
    values_dict = {'CE':
    {'MNIST':{'FashionMNIST':[],'KMNIST':[]},
    'FashionMNIST':{'MNIST':[],'KMNIST':[]},
    'KMNIST':{'MNIST':[],'FashionMNIST':[]},
    'CIFAR10':{'SVHN':[],'CIFAR100':[]},
    'CIFAR100':{'SVHN':[],'CIFAR10':[]}
    },
    
    'Moco':
    {'MNIST':{'FashionMNIST':[],'KMNIST':[]},
    'FashionMNIST':{'MNIST':[],'KMNIST':[]},
    'KMNIST':{'MNIST':[],'FashionMNIST':[]},
    'CIFAR10':{'SVHN':[],'CIFAR100':[]},
    'CIFAR100':{'SVHN':[],'CIFAR10':[]}
    },

    'SupCon':
    {'MNIST':{'FashionMNIST':[],'KMNIST':[]},
    'FashionMNIST':{'MNIST':[],'KMNIST':[]},
    'KMNIST':{'MNIST':[],'FashionMNIST':[]},
    'CIFAR10':{'SVHN':[],'CIFAR100':[]},
    'CIFAR100':{'SVHN':[],'CIFAR10':[]}
    },
    }
    api = wandb.Api()
    data_array = np.empty((10,3))
    data_array[:] = np.nan
    # Array for std values
    std_array = np.empty((10,3))
    std_array[:] = np.nan
    
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats"})
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

        name_list.append(run.name)

        keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
        auroc_keys = [key for key in keys if 'table' not in key.lower()]
        # Remove keys which use emnist 
        auroc_keys = [key for key in auroc_keys if 'emnist' not in key.lower()]
        for key in auroc_keys:
            OOD_dataset = ood_dataset_string(key, dataset_dict, ID_dataset)
            # Different way to obtain the auroc value based on whether the value is 
            auroc_value = summary_list[i][key][-1] if desired_key == 'Marginal Typicality Entropy Average Likelihood Batch Size'.lower() else summary_list[i][key]
            # Append the values to a list
            values_dict[model_type][ID_dataset][OOD_dataset].append(auroc_value)
    
    # All values to iterate through
    Models = ['CE','Moco','SupCon']    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100']
    all_OOD = {'MNIST':['FashionMNIST','KMNIST'],
    'FashionMNIST':['MNIST','KMNIST'],
    'KMNIST':['MNIST','FashionMNIST'],
    'CIFAR10':['SVHN','CIFAR100'],
    'CIFAR100':['SVHN','CIFAR10']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            for OOD in all_OOD[ID]: # Go through the different OOD datasets for a particular ID dataset
                column = key_dict['model_type'][Model]
                row = 2*key_dict['dataset'][ID] + dataset_dict[ID][OOD]
                mean_val = np.around(np.mean(values_dict[Model][ID][OOD]),decimals=decimal_places) 
                std_val = np.around(np.std(values_dict[Model][ID][OOD]),decimals=decimal_places)
                data_array[row, column] = mean_val
                std_array[row, column] = std_val

    # Obtain the names of the rows and the name of the columns
    #column_names = [model for model in key_dict['model_type'].keys()]
    column_names= ['SupCLR' if model=='SupCon' else model for model in key_dict['model_type'].keys()]
    row_names = []
    # iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
    for dataset in key_dict['dataset'].keys():
        for OOD_dataset in dataset_dict[dataset].keys():
            row_names.append(f'ID:{dataset}, OOD:{OOD_dataset}')
    
    auroc_mean_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
    auroc_std_df = pd.DataFrame(std_array, columns = column_names, index = row_names)

    auroc_mean_latex_table = auroc_mean_df.to_latex()
    auroc_std_latex_table =  auroc_std_df.to_latex()
    


    mean_values_string = re.findall("\d+\.\d+", auroc_mean_latex_table)
    std_values_string = re.findall("\d+\.\d+", auroc_std_latex_table)
    
    concat_list_string = [mean_values_string[i] + ' \pm ' + std_values_string[i] for i in range(len(mean_values_string))]
    
    concatenated_list = []
    recursive_string = copy.copy(auroc_mean_latex_table)
    for i in range(len(mean_values_string)):
        first_string, recursive_string = recursive_string.split(mean_values_string[i],1) 
        first_string = first_string + mean_values_string[i]
        first_string = first_string.replace(mean_values_string[i], concat_list_string[i])
        concatenated_list.append(first_string)

        #import ipdb ; ipdb.set_trace()

        #auroc_mean_latex_table = auroc_mean_latex_table.replace(mean_values_string[i], concat_list_string[i])

    # Concenate the different parts with no space in between 
    auroc_mean_latex_table = ''.join(concatenated_list)

    # Table post processing
    auroc_mean_latex_table = auroc_mean_latex_table.replace('{}','{Datasets}')
    auroc_mean_latex_table = auroc_mean_latex_table.replace("lrrr","|p{3cm}|p{1.25cm}|p{1.25cm}|p{1.25cm}|")
    auroc_mean_latex_table = auroc_mean_latex_table.replace(r"\toprule",r"\hline")
    auroc_mean_latex_table = auroc_mean_latex_table.replace(r"\\",r"\\ \hline")
    auroc_mean_latex_table = auroc_mean_latex_table.replace(r"\midrule"," ")
    auroc_mean_latex_table = auroc_mean_latex_table.replace(r"\bottomrule"," ")

    auroc_mean_latex_table = auroc_mean_latex_table + ' \\\ \hline'
    print(auroc_mean_latex_table)
    import ipdb; ipdb.set_trace()
if __name__ == '__main__':
    ood_detection_repeat_table()
