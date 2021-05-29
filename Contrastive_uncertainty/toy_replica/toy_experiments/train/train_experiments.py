# Import parameters for different training methods


from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.config.hsup_con_td_params import hsup_con_td_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.config.hsup_con_bu_centroid_params import hsup_con_bu_centroid_hparams


# Importing the different lightning modules for the baselines

from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_module import HSupConBUToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.models.hsup_con_td_module import HSupConTDToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroidToy


# Model instances for the different methods
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_model_instance import ModelInstance as HSupConBUModelInstance
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.models.hsup_con_td_model_instance import ModelInstance as HSupConTDModelInstance
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance as HSupConBUCentroidModelInstance


# Import training methods 
from Contrastive_uncertainty.toy_replica.toy_general.train.train_general import train as general_training
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.train_general_hierarchy import train as general_hierarchy_training

def train(base_dict):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon',
    'PCL','MultiPCL','UnSupConMemory','HSupCon','HSupConBU','HSupConBUCentroid','HSupConTD']

    # Dict for the model name, parameters and specific training loop
    model_dict = {'HSupConBUCentroid':{'params':hsup_con_bu_centroid_hparams,'model_module':HSupConBUCentroidToy, 
                    'model_instance':HSupConBUCentroidModelInstance, 'train':general_hierarchy_training},
                    
                    'HSupConBU':{'params':hsup_con_bu_hparams,'model_module':HSupConBUToy, 
                    'model_instance':HSupConBUModelInstance,'train':general_hierarchy_training},

                    'HSupConTD':{'params':hsup_con_td_hparams,'model_module':HSupConTDToy, 
                    'model_instance':HSupConTDModelInstance,'train':general_hierarchy_training},
          
    }
    # Update the parameters of each model

    # iterate through all items of the state dict
    for base_k, base_v in base_dict.items():
        # Iterate through all the model dicts
        for model_k, model_v in model_dict.items():
            # Go through each dict one by one and check if base k in model params
            if base_k in model_dict[model_k]['params']:
                # update model key with base params
                model_dict[model_k]['params'][base_k] = base_v


    # Checks whether base_dict single model is present in the list
    assert base_dict['single_model'] in acceptable_single_models, 'single model response not in list of acceptable responses'
    
    datasets = ['Blobs','Blobs']
    ood_datasets = ['TwoMoons','Blobs']
    
    # BASELINES
    # Go through all the models in the current dataset and current OOD dataset
    if base_dict['single_model']== 'Baselines':
        for model_k, model_v in model_dict.items():
            params = model_dict[model_k]['params']
            train_method = model_dict[model_k]['train']
            model_module = model_dict[model_k]['model_module'] 
            model_instance_method = model_dict[model_k]['model_instance']
            # Try statement to allow the code to continue even if a single run fails
            train_method(params, model_module, model_instance_method)


    ## SINGLE MODEL
    # Go through a single model on all different datasets
    else:
        # Name of the chosen model
        chosen_model = base_dict['single_model']
        # Specific model dictionary chosen
        model_info = model_dict[chosen_model]
        train_method = model_info['train']
        params = model_info['params']
        model_module = model_info['model_module'] 
        model_instance_method = model_info['model_instance']
        # Loop through the different datasets and OOD datasets and examine if the model is able to train for the task
        for dataset, ood_dataset in zip(datasets, ood_datasets):
            params['dataset'] = dataset
            params['OOD_dataset'] = ood_dataset
            train_method(params, model_module, model_instance_method)
            