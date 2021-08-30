# Plotting the centroid distances against the confuson log probability

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import general params
import json

from Contrastive_uncertainty.experiments.run.report_analysis.centroid_clp_plot_analysis import class_confusion_log_probability_vector
from Contrastive_uncertainty.experiments.run.report_analysis.centroid_total_kl_plot_analysis import total_kl_div_vector


# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplot.html 
# Make it so that the model can loop for the different datasets as well as different models
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
def total_kl_clp_subplot():
    # Desired ID,OOD and Model
    #Models = ['Moco','SupCon']
    #Models = ['SupCon']
    Models = ['SupCon']

    all_ID = ['MNIST','FashionMNIST','CIFAR10']
    all_OOD = {'MNIST':['FashionMNIST'],
    'FashionMNIST':['MNIST'],
    'CIFAR10':['SVHN','CIFAR100']}
    '''
    Models = ['Moco']
    
    all_ID = ['MNIST','FashionMNIST','CIFAR100']
    all_OOD = {'MNIST':['FashionMNIST'],
    'FashionMNIST':['MNIST'],
    'CIFAR100':['SVHN','CIFAR10']}
    '''
    #https://stackoverflow.com/questions/14770735/how-do-i-change-the-figure-size-with-subplots
    plt.figure(figsize=(12,12)) # Changing size of the figure before I make it into subplots
    subplot_num = 1
    for Model in Models:# Go through the different models
        for ID in all_ID: # Go through the different ID dataset
            for OOD in all_OOD[ID]: # Go through the different OOD datasets for a particular ID dataset        
                api = wandb.Api()
                # Gets the runs corresponding to a specific filter
                # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


                # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
                runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]}) # only look at the runs related to Moco and SupCon
                summary_list, config_list, name_list = [], [], []

    

                root_dir = 'run_data/'
                for i, run in enumerate(runs): 
                    print('run:',i)
                    # .summary contains the output keys/values for metrics like accuracy.
                    #  We call ._json_dict to omit large files 
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
                    # Obtain the dataset and the model type
                    run_dir = root_dir + run_path
                    # Read dir is how to read the file
                    read_dir = run_dir + '/' + data_kl_dir

                    dataset = config_list[i]['dataset']
                    model_type = config_list[i]['model_type']
                    if ID == dataset and Model == model_type:
                        with open(read_dir) as f:
                            total_kl_data = json.load(f)

                        # Calculate the mean distance
                        total_kl_values = total_kl_div_vector(total_kl_data)
                        break # To stop the loop

                    
                clp_runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})
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
                        collated_array = np.stack((total_kl_values,clp_values),axis=1)  
                        df = pd.DataFrame(collated_array)
                        columns = ['KL(Total||Class) (Nats)', 'CLP']
                        df.columns = columns
                        
                        
                        fit = np.polyfit(df['KL(Total||Class) (Nats)'], df['CLP'], 1)
                        plt.subplot(4, 1, subplot_num)
                        sns.regplot(x = df['KL(Total||Class) (Nats)'], y = df['CLP'],color='blue')
                        #plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.90), xycoords='axes fraction')
                        plt.title(f'{ID}-{OOD}', size=12)
                        subplot_num +=1
                        print('subplot_num:',subplot_num)
        # Adjust position of the subplots https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/#:~:text=We%20can%20use%20the%20plt,axis%20width%20and%20height%2C%20respectively.
        plt.subplots_adjust(left=0.15,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
        #plt.show()
        

        folder = f'Scatter_Plots/KL_CLP_plots/{model_type}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f'{folder}/KL_CLP_{Model_name}_collated.png')
        #import ipdb; ipdb.set_trace()
        plt.close()
                        
                        #fit = np.polyfit(df['KL(Total||Class) (Nats)'], df['CLP'], 1)
                        #fig = plt.figure(figsize=(10, 7))
                        #sns.regplot(x = df['KL(Total||Class) (Nats)'], y = df['CLP'],color='blue')
                        #plt.annotate('y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), xy=(0.05, 0.95), xycoords='axes fraction')
                        ##plt.text(3.2, -7.12, 'y={:.2f}+{:.2f}*x'.format(fit[1], fit[0]), color='darkblue', size=12)
                        #plt.title(f'KL Divergence and Confusion Log Probability for {ID}-{OOD} pair using {Model_name} model', size=12)
                        ## regression equations
                        #
                        #folder = f'Scatter_Plots/KL_CLP_plots/{model_type}'
                        #
#
                        #if not os.path.exists(folder):
                        #    os.makedirs(folder)
                        #plt.savefig(f'{folder}/KL_CLP_{ID}_{OOD}_{Model_name}.png')
                        #plt.close()
                        
if __name__ == '__main__':
    total_kl_clp_subplot()