# Plotting the centroid distances against the confuson log probability

import ipdb
from numpy.lib.function_base import percentile
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

from Contrastive_uncertainty.experiments.run.report_analysis.centroid_clp_plot_analysis import class_centroid_distance_vector

def overlapping_fraction_vector(json_data,percentile):
    data = np.array(json_data['data'])
    overlapping_values = np.around(data[:,0],decimals=4) if percentile=='lower' else np.around(data[:,0],decimals=4)
    return overlapping_values


def overlapping_fraction_saving():
    desired_key = 'Non Class Percentage Overlap'
    desired_key = desired_key.lower()
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})
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

#overlapping_fraction_saving()


def centroid_overlapping_fraction_plot():
    
    percentile = 'lower' #  whether to get the upper or lower percentile

    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.
    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"}) # only look at the runs related to Moco and SupCon
    
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
        data_overlapping_dir = summary_list[i]['Non Class Percentage Overlap']['path']
        run_dir = root_dir + run_path
        # Read dir is how to read the file
        read_dir = run_dir + '/' + data_overlapping_dir
        
        
        data_centroid_dir = summary_list[i]['Centroid Distances Average vector_table']['path']  
        
        # Obtain the dataset and the model type
        dataset = config_list[i]['dataset']
        model_type = config_list[i]['model_type']
        Model_name = 'SupCLR' if model_type =='SupCon' else model_type
        with open(data_centroid_dir) as f: 
            centroid_data = json.load(f)

        with open(read_dir) as f:
            overlapping_fraction_data = json.load(f)
            # Calculate the mean distance
        
        class_centroid_distances = class_centroid_distance_vector(centroid_data)
        overlapping_fraction = overlapping_fraction_vector(overlapping_fraction_data,percentile)

        collated_array = np.stack((class_centroid_distances,overlapping_fraction),axis=1)
        df = pd.DataFrame(collated_array)
        columns = ['Class Centroid Distance', f'Overlapping fraction {percentile} percentile']
        df.columns = columns
        fit = np.polyfit(df['Class Centroid Distance'], df[f'Overlapping fraction {percentile} percentile'], 1)

        fig = plt.figure(figsize=(10, 7))
        sns.regplot(x = df['Class Centroid Distance'], y = df[f'Overlapping fraction {percentile} percentile'],color='blue')
        plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
        plt.title(f'Class Centroid Distance and Overlapping fraction: {percentile} percentile for {dataset} dataset using {Model_name} model', size=12)
        plt.ylim(bottom=0.0, top=1.0)
        folder = f'Scatter_Plots/Centroid_Distance_Overlapping_Fraction_plots/{percentile}/{model_type}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        plt.savefig(f'{folder}/Class_centroid_overlapping_fraction_{percentile}_{dataset}_{Model_name}.png')
        plt.close()

if __name__ == '__main__':
    centroid_overlapping_fraction_plot()