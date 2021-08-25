# Plotting the centroid distances against the confuson log probability

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

from Contrastive_uncertainty.experiments.run.report_analysis.centroid_clp_plot_analysis import class_centroid_distance_vector



def total_kl_div_vector(json_data):
    data = np.array(json_data['data'])
    kl_values = np.around(data[:,0],decimals=3)
    return kl_values


def total_kl_saving():
    desired_key = 'kl divergence(total||class)'
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    #runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon

    summary_list, config_list, name_list = [], [], []
    # Change the root directory to save the file in the total KL divergence section
    root_dir = 'run_data/'
    for i, run in enumerate(runs): 
        #import ipdb; ipdb.set_trace()
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)
        run_path = '/'.join(runs[i].path)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        ID_dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']

        # .name is the human-readable name of the run.dir
        name_list.append(run.name)
        keys = [key for key, value in summary_list[i].items() if desired_key in key.lower()]
        
        for key in keys:    
            data_dir = summary_list[i][key]['path'] 
            run_dir = root_dir + run_path
            file_data = json.load(run.file(data_dir).download(root=run_dir))

#total_kl_saving()



def centroid_total_kl_plot():
    
    # Desired ID,OOD and Model
    
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.
    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon
    
    summary_list, config_list, name_list = [], [], []
    # Dict to map distances of specific datasets and model types to the data array
    
    root_dir = 'run_data/'
    for i, run in enumerate(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        values = run.summary
        summary_list.append(run.summary._json_dict)
        run_path = '/'.join(runs[i].path)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.dir
        name_list.append(run.name)
        data_kl_dir = summary_list[i]['KL Divergence(Total||Class)']['path']
        run_dir = root_dir + run_path
        # Read dir is how to read the file
        read_dir = run_dir + '/' + data_kl_dir
        
        data_centroid_dir = summary_list[i]['Centroid Distances Average vector_table']['path']  
        
        # Obtain the dataset and the model type
        dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']
        Model_name = 'SupCLR' if model_type=='SupCon' else model_type
        with open(data_centroid_dir) as f: 
            centroid_data = json.load(f) 
            
        with open(read_dir) as f:
            total_kl_data = json.load(f)
            # Calculate the mean distance
        
        class_centroid_distances = class_centroid_distance_vector(centroid_data)
        total_kl_values = total_kl_div_vector(total_kl_data)

        collated_array = np.stack((class_centroid_distances,total_kl_values),axis=1)
        df = pd.DataFrame(collated_array)
        columns = ['Class Centroid Distance', 'KL(Total||Class) (Nats)']
        df.columns = columns
        fit = np.polyfit(df['Class Centroid Distance'], df['KL(Total||Class) (Nats)'], 1)
        fig = plt.figure(figsize=(10, 7))
        sns.regplot(x = df['Class Centroid Distance'], y = df['KL(Total||Class) (Nats)'],color='blue')
        plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
        plt.title(f'Class Centroid Distance and KL Divergence for {dataset} dataset using {Model_name} model', size=12)
        plt.ylim(0)
        '''
        if model_type =='Moco':
            plt.xlim(2.0, 6.0)
            plt.ylim(0, 1000)
        else:
            plt.xlim(8.0, 12.5)
            plt.ylim(0, 5000)
        '''
        #plt.show()

        folder = f'Scatter_Plots/Centroid_Distance_KL_plots/{model_type}'
        if not os.path.exists(folder):
            #https://www.geeksforgeeks.org/python-os-makedirs-method/#:~:text=makedirs()%20method%20in%20Python,method%20will%20create%20them%20all.&text=Suppose%20we%20want%20to%20create,are%20unavailable%20in%20the%20path.
            # make recursive folders
            os.makedirs(folder)
        plt.savefig(f'{folder}/Class_centroid_KL_{dataset}_{Model_name}.png')
        plt.close()

if __name__ == '__main__':
    centroid_total_kl_plot()