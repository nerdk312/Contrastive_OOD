# Plotting the centroid distances against the confuson log probability

import ipdb
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json


# Calculates the mean vector distance from the individual classes distances
def class_centroid_distance_vector(json_data):
    data = np.array(json_data['data'])
    # Obtain the distances only for the different classes
    distances  = np.around(data[:,1],decimals= 3)
    return distances
    
# Make it so that the model can loop for the different datasets as well as different models
def centroid_clp_plot():
    # Desired ID,OOD and Model
    ID = 'MNIST'
    OOD = 'FashionMNIST'
    Model ='Moco'
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
            clp_values = clp_summary_list[i][desired_key]
            clp_values = np.array(clp_values)
            #import ipdb; ipdb.set_trace()
            collated_array = np.stack((class_centroid_distances,clp_values),axis=1) 
            df = pd.DataFrame(collated_array)
            columns = ['Class Centroid Distance', 'CLP']
            df.columns = columns

            fit = np.polyfit(df['Class Centroid Distance'], df['CLP'], 1)
            #import ipdb; ipdb.set_trace()
            fig = plt.figure(figsize=(10, 7))
            sns.regplot(x = df['Class Centroid Distance'], y = df['CLP'],color='blue')
            plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
            #plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
            plt.title(f'Class Centroid Distance and Confusion Log Probability for {ID}-{OOD} pair using {Model} model', size=12)
            # regression equations
            
            folder = 'Scatter_Plots'
            if not os.path.exists(folder):
                os.mkdir(folder)
            
            plt.savefig(f'{folder}/Class_centroid_CLP_{ID}_{OOD}_{Model}.png')
# 
    # '''
    #Calculating the name of the rows and the columns
    # column_names =  [dataset for dataset in key_dict['dataset'].keys()] 
    # row_names = [model for model in key_dict['model_type'].keys()]
    # '''
    # row_names =  [dataset for dataset in key_dict['dataset'].keys()] 
    # column_names = [model for model in key_dict['model_type'].keys()]
# 
    #making the dataframe and with the specified rows and columns, then converting it into a latex table
    # mean_vector_df = pd.DataFrame(data_array, columns=column_names,index=row_names)
    # latex_table = mean_vector_df.to_latex()
# 
    # latex_table = latex_table.replace('{}','{Dataset}')
    # latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
    # latex_table = latex_table.replace(r"\toprule",r"\hline")
    # latex_table = latex_table.replace(r"\midrule"," ")
    # latex_table = latex_table.replace(r"\bottomrule"," ")
    #latex_table = latex_table.replace(r"\midrule",r"\hline")
    #latex_table = latex_table.replace(r"\bottomrule",r"\hline")
    #https://stackoverflow.com/questions/24704299/how-to-treat-t-as-a-regular-string-in-python
# 
    # latex_table = latex_table.replace(r'\\',r'\\ \hline')
    # print(latex_table)
# 

centroid_clp_plot()