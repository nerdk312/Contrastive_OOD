# Script used to get the fraction of OOD samples in the K classes closest to the total training centroid

import wandb
import pandas as pd
import numpy as np

# Import general params
import json
import math


# For each ID dataset, it maps the dict to another value
dataset_dict = {'MNIST': {'FashionMNIST':0, 'KMNIST':1},
            'FashionMNIST': {'MNIST':0, 'KMNIST':1},
            'KMNIST': {'MNIST':0, 'FashionMNIST':1},
            'CIFAR10': {'SVHN':0, 'CIFAR100':1},
            'CIFAR100': {'SVHN':0, 'CIFAR10':1}
}
# Dict for the specific case to the other value
key_dict = {'model_type':{'CE':0, 'Moco':1, 'SupCon':2},
            'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}

# Used to calculate the fraction of OOD samples in the classes which are in the K bottom values
def OOD_fraction_calculation(centroid_json,auroc_json):
    centroid_data = np.array(centroid_json['data'])
    # k controls how many values I choose
    k_values = 2
    bottom_k = bottom_k_values(centroid_data,k_values)    
    auroc_data = np.array(auroc_json['data'])

    OOD_sum = 0
    # Iterate through the different values in bottom k 1d array
    for i in bottom_k:
        OOD_fraction = float(auroc_data[i][-1])  # Get the index and then get the last value which corresponds to the value for the OOD fraction
        OOD_sum += OOD_fraction

    return OOD_sum

# Calculate the bottom k class centroid indices
def bottom_k_values(centroid_data, k_values):
    # Calculate the bottom K values
    bottom_k = []
    distances = centroid_data[:,1]
    for k in range(k_values):
        bottom_index = np.argmin(distances)
        distances[bottom_index] = math.inf  # Change the value to infinity to prevent it from using the next value
        bottom_k.append(bottom_index)
    
    return np.array(bottom_k)


# Check if ood_dataset substring is present in string
def ood_dataset_string(key, dataset_dict, ID_dataset):
    split_keys = key.lower().split() # Make the key lower and then split the string at locations where is a space
    OOD_dict = dataset_dict[ID_dataset]
    for key in OOD_dict.keys():
        if key.lower() in split_keys:
            ood_dataset = key

    return ood_dataset




def obtain_table():   
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})


    summary_list, config_list, name_list = [], [], []

    # 15 different simulations and each simulation has 2 OOD datasets associated with it
    #data_array = np.zeros((30,3)) # potentially could make the shape (3,30)
    data_array = np.empty((10,3))
    data_array[:] = np.nan

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

        # Only a single directory required for the module
        centroid_data_dir = summary_list[i]['Centroid Distances Average vector_table']['path'] 
        # Obtain the dataset and the model type
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
                auroc_data = json.load(f) 

            with open(centroid_data_dir) as g:
                centroid_data = json.load(g)
            # Obtain the data
            OOD_fraction = OOD_fraction_calculation(centroid_data, auroc_data)
            # Inputting data
            column = key_dict['model_type'][model_type]
            row = 2*key_dict['dataset'][ID_dataset] + dataset_dict[ID_dataset][OOD_dataset]
            data_array[row, column] = OOD_fraction


    # Obtain the names of the rows and the name of the columns
    column_names = [model for model in key_dict['model_type'].keys()]

    row_names = []
    # iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
    for dataset in key_dict['dataset'].keys():
        for OOD_dataset in dataset_dict[dataset].keys():
            row_names.append(f'ID:{dataset}, OOD:{OOD_dataset}')


    OOD_fraction_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
    latex_table = OOD_fraction_df.to_latex()

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


'''
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})


summary_list, config_list, name_list = [], [], []

# 15 different simulations and each simulation has 2 OOD datasets associated with it
#data_array = np.zeros((30,3)) # potentially could make the shape (3,30)
data_array = np.empty((10,3))
data_array[:] = np.nan

# For each ID dataset, it maps the dict to another value
dataset_dict = {'MNIST': {'FashionMNIST':0, 'KMNIST':1},
            'FashionMNIST': {'MNIST':0, 'KMNIST':1},
            'KMNIST': {'MNIST':0, 'FashionMNIST':1},
            'CIFAR10': {'SVHN':0, 'CIFAR100':1},
            'CIFAR100': {'SVHN':0, 'CIFAR10':1}

}
# Dict for the specific case to the other value
key_dict = {'model_type':{'CE':0, 'Moco':1, 'SupCon':2},
            'dataset': {'MNIST':0, 'FashionMNIST':1, 'KMNIST':2, 'CIFAR10':3,'CIFAR100':4}}
                        

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

    # Only a single directory required for the module
    centroid_data_dir = summary_list[i]['Centroid Distances Average vector_table']['path'] 
    # Obtain the dataset and the model type
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
            auroc_data = json.load(f) 
        
        with open(centroid_data_dir) as g:
            centroid_data = json.load(g)
        # Obtain the data
        OOD_fraction = OOD_fraction_calculation(centroid_data, auroc_data)
        # Inputting data
        column = key_dict['model_type'][model_type]
        row = 2*key_dict['dataset'][ID_dataset] + dataset_dict[ID_dataset][OOD_dataset]
        data_array[row, column] = OOD_fraction


# Obtain the names of the rows and the name of the columns
column_names = [model for model in key_dict['model_type'].keys()]

row_names = []
# iterate through the ID dataset, and iterate for all the OOD datasets in the ID dataset
for dataset in key_dict['dataset'].keys():
    for OOD_dataset in dataset_dict[dataset].keys():
        row_names.append(f'ID:{dataset}, OOD:{OOD_dataset}')


OOD_fraction_df = pd.DataFrame(data_array, columns = column_names, index = row_names)
latex_table = OOD_fraction_df.to_latex()

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
'''