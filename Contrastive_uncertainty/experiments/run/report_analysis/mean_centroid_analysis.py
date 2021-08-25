import wandb
import pandas as pd
import numpy as np

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams
import json

# Calculates the mean vector distance from the individual classes distances
def mean_vector_calculation(json_data):
    data = np.array(json_data['data'])
    # Obtain the distances only for the different classes
    distances  = data[:,1]
    # Calculate the mean and round to a certain number of decimal places
    mean_distance = np.mean(distances)
    mean_distance = np.around(mean_distance, decimals=3)
    return mean_distance
    

def centroid_table():
    api = wandb.Api()
    # Gets the runs corresponding to a specific filter
    # https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


    # Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"})


    summary_list, config_list, name_list = [], [], []

    #data_array = np.zeros((3,5)) # potentially could make the shape (5,3)
    data_array = np.zeros((5,3)) # potentially could make the shape (5,3)


    # Dict to map distances of specific datasets and model types to the data array
    key_dict = {'dataset':{'MNIST':0, 'FashionMNIST':1,'KMNIST':2, 'CIFAR10':3, 'CIFAR100':4},
                'model_type':{'CE':0, 'Moco':1, 'SupCon':2}}


    for i, run in enumerate(runs): 
        #import ipdb; ipdb.set_trace()
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

        with open(data_dir) as f: 
            data = json.load(f) 

        # Calculate the mean distance
        mean_distance = mean_vector_calculation(data)

        # Choose the row and column of the data based on the specific value present
        '''
        row = key_dict['model_type'][model_type]
        column = key_dict['dataset'][dataset]
        '''
        column = key_dict['model_type'][model_type]
        row = key_dict['dataset'][dataset]

        data_array[row, column] = mean_distance

    '''
    # Calculating the name of the rows and the columns
    column_names =  [dataset for dataset in key_dict['dataset'].keys()] 
    row_names = [model for model in key_dict['model_type'].keys()]
    '''
    row_names =  [dataset for dataset in key_dict['dataset'].keys()] 
    column_names = [model for model in key_dict['model_type'].keys()]

    # making the dataframe and with the specified rows and columns, then converting it into a latex table
    mean_vector_df = pd.DataFrame(data_array, columns=column_names,index=row_names)
    latex_table = mean_vector_df.to_latex()

    latex_table = latex_table.replace('{}','{Dataset}')
    latex_table = latex_table.replace("lrrr","|p{3cm}|c|c|c|")
    latex_table = latex_table.replace(r"\toprule",r"\hline")
    latex_table = latex_table.replace(r"\midrule"," ")
    latex_table = latex_table.replace(r"\bottomrule"," ")
    #latex_table = latex_table.replace(r"\midrule",r"\hline")
    #latex_table = latex_table.replace(r"\bottomrule",r"\hline")
    #https://stackoverflow.com/questions/24704299/how-to-treat-t-as-a-regular-string-in-python

    latex_table = latex_table.replace(r'\\',r'\\ \hline')
    print(latex_table)
if __name__== '__main__':
    centroid_table()
