# Plotting the centroid distances against the confuson log probability

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

from Contrastive_uncertainty.experiments.run.report_analysis.analysis_utils import generic_saving

# Calculates the mean vector distance from the individual classes distances
def class_centroid_distance_vector(json_data):
    data = np.array(json_data['data'])
    # Obtain the distances only for the different classes
    distances  = np.around(data[:,1],decimals= 3)
    return distances

def class_confusion_log_probability_vector(json_data):
    data = np.array(json_data['data'])
    clp_values = np.around(data[:,0],decimals=3)
    return clp_values
    
# Make it so that the model can loop for the different datasets as well as different models
def centroid_clp_plot():
    # Desired ID,OOD and Model
    Models = ['CE','Moco','SupCon']
    #Models = ['SupCon']
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10']
    all_OOD = {'MNIST':['FashionMNIST','KMNIST'],
    'FashionMNIST':['MNIST','KMNIST'],
    'KMNIST':['MNIST','FashionMNIST'],
    'CIFAR10':['SVHN','CIFAR100']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            for OOD in all_OOD[ID]: # Go through the different OOD datasets for a particular ID dataset        
                api = wandb.Api()
                # Gets the runs corresponding to a specific filter
                # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


                # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
                runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})
                summary_list, config_list, name_list = [], [], []

                # Dict to map distances of specific datasets and model types to the data array
                key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4},
                            'model_type':{'CE':0, 'Moco':1, 'SupCon':2}}


                for i, run in enumerate(runs): 
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
                    data_dir = summary_list[i]['Centroid Distances Average vector_table']['path']    
                    # Obtain the dataset and the model type
                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    if ID == dataset and Model == model_type:
                        with open(data_dir) as f: 
                            data = json.load(f) 

                        # Calculate the mean distance
                        class_centroid_distances = class_centroid_distance_vector(data)

                        break # To stop the loop

                    
                clp_runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})
                desired_key = 'Class Wise CLP'
                clp_summary_list, clp_config_list, clp_name_list = [], [], []

                for i, clp_run in enumerate(clp_runs): 

                    # .summary contains the output keys/values for metrics like accuracy.
                        #  We call ._json_dict to omit large files 
                    clp_summary_list.append(clp_run.summary._json_dict)

                    # .config contains the hyperparameters.
                    #  We remove special values that start with _.
                    clp_config_list.append(
                        {k: v for k,v in clp_run.config.items()
                         if not k.startswith('_')})

                    ID_dataset = clp_config_list[i]['dataset']
                    OOD_dataset = clp_config_list[i]['OOD_dataset'][0]
                    if ID == ID_dataset and OOD == OOD_dataset:

                        # Take into account the difference in the name of the model for SupCon and SupCLR
                        Model_name = 'SupCLR' if Model=='SupCon' else Model
                        
                        clp_values = clp_summary_list[i][desired_key]
                        clp_values = np.array(clp_values)
                        collated_array = np.stack((class_centroid_distances,clp_values),axis=1) 
                        df = pd.DataFrame(collated_array)
                        columns = ['Class Centroid Distance', 'CLP']
                        df.columns = columns

                        fit = np.polyfit(df['Class Centroid Distance'], df['CLP'], 1)
                        fig = plt.figure(figsize=(10, 7))
                        sns.regplot(x = df['Class Centroid Distance'], y = df['CLP'],color='blue')
                        plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
                        #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                        plt.title(f'Class Centroid Distance and Confusion Log Probability for {ID}-{OOD} pair using {Model_name} model', size=12)
                        # regression equations

                        folder = f'Scatter_Plots/{Model}'

                        if not os.path.exists(folder):
                            os.mkdir(folder)
                        

                        plt.savefig(f'{folder}/Class_centroid_CLP_{ID}_{OOD}_{Model_name}.png')
                        plt.close()



# Make it so that the model can loop for the different datasets as well as different models
def centroid_clp_plot_v2():
    # Desired ID,OOD and Model
    Models = ['CE','Moco','SupCon']
    #Models = ['SupCon']
    
    all_ID = ['MNIST','FashionMNIST','KMNIST', 'CIFAR10','CIFAR100']
    all_OOD = {'MNIST':['FashionMNIST','KMNIST'],
    'FashionMNIST':['MNIST','KMNIST'],
    'KMNIST':['MNIST','FashionMNIST'],
    'CIFAR10':['SVHN','CIFAR100'],
    'CIFAR100':['SVHN','CIFAR10']}

    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            for OOD in all_OOD[ID]: # Go through the different OOD datasets for a particular ID dataset        
                api = wandb.Api()
                # Gets the runs corresponding to a specific filter
                # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


                # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
                runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})
                summary_list, config_list, name_list = [], [], []

                # Dict to map distances of specific datasets and model types to the data array
                key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4},
                            'model_type':{'CE':0, 'Moco':1, 'SupCon':2}}


                for i, run in enumerate(runs): 
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
                    data_dir = summary_list[i]['Centroid Distances Average vector_table']['path']    
                    # Obtain the dataset and the model type
                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    if ID == dataset and Model == model_type:
                        with open(data_dir) as f: 
                            data = json.load(f) 

                        # Calculate the mean distance
                        class_centroid_distances = class_centroid_distance_vector(data)

                        break # To stop the loop

                    
                clp_runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})
                #desired_key = 'Class Wise Confusion Log Probability'
                #desired_key = desired_key.lower()
                clp_summary_list, clp_config_list, clp_name_list = [], [], []
                root_dir = 'run_data/'
                for i, clp_run in enumerate(clp_runs): 

                    # .summary contains the output keys/values for metrics like accuracy.
                        #  We call ._json_dict to omit large files 
                    clp_summary_list.append(clp_run.summary._json_dict)

                    # .config contains the hyperparameters.
                    #  We remove special values that start with _.
                    clp_config_list.append(
                        {k: v for k,v in clp_run.config.items()
                         if not k.startswith('_')})

                    ID_dataset = clp_config_list[i]['dataset']
                    OOD_dataset = clp_config_list[i]['OOD_dataset'][0]
                    group_name = clp_config_list[i]['group']
                    if ID == ID_dataset and OOD == OOD_dataset:

                        # Take into account the difference in the name of the model for SupCon and SupCLR
                        Model_name = 'SupCLR' if Model=='SupCon' else Model
                        path_list = clp_runs[i].path
                        path_list.insert(-1, group_name) # insert the group name in the location one before the last value (rather than the last value which is peculiar)
                        run_path = '/'.join(path_list)
                        run_dir = root_dir + run_path
                        clp_values_dir = clp_summary_list[i]['Class Wise Confusion Log Probability']['path']
                        read_dir = run_dir + '/' + clp_values_dir

                        with open(read_dir) as f:
                            class_wise_clp_json = json.load(f)
                        
                        clp_values = class_confusion_log_probability_vector(class_wise_clp_json)
                        collated_array = np.stack((class_centroid_distances,clp_values),axis=1)  
                        df = pd.DataFrame(collated_array)
                        columns = ['Class Centroid Distance', 'CLP']
                        df.columns = columns

                        fit = np.polyfit(df['Class Centroid Distance'], df['CLP'], 1)
                        fig = plt.figure(figsize=(10, 7))
                        sns.regplot(x = df['Class Centroid Distance'], y = df['CLP'],color='blue')
                        plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
                        #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                        plt.title(f'Class Centroid Distance and Confusion Log Probability for {ID}-{OOD} pair using {Model_name} model', size=12)
                        # regression equations
                        
                        folder = f'Scatter_Plots/Centroid_CLP_plots/{model_type}'
                        

                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f'{folder}/Class_centroid_CLP_{ID}_{OOD}_{Model_name}.png')
                        plt.close()
                        
if __name__ == '__main__':
    centroid_clp_plot_v2()
#           
#centroid_clp_plot()





'''
desired_key = 'Class Wise Confusion Log Probability'
run_filter = {"config.group":"Confusion Log Probability Evaluation"}
generic_saving(desired_key, run_filter)
'''