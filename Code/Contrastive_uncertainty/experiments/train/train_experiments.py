# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams


from Contrastive_uncertainty.ensemble.config.cross_entropy_ensemble_params import cross_entropy_ensemble_hparams


# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.sup_con.models.sup_con_module import SupConModule
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_module import CrossEntropyEnsembleModule



# Model instances for the different methods
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.sup_con.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_model_instance import ModelInstance as CEEnsembleModelInstance


# Import datamodule info
from Contrastive_uncertainty.general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict


# Import training methods 
from Contrastive_uncertainty.general.train.train_general import train as general_training

from Contrastive_uncertainty.general.train.train_general_confusion import train as general_confusion_training

def train(base_dict):    
    acceptable_single_models = ['Baselines',
    #'CE',
    #'Moco',
    #'SupCon',
    'CEEnsemble'
    ]

    # Dict for the model name, parameters and specific training loop
    
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyModule,
                    'model_instance':CEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},
        
                    'Moco':{'params':moco_hparams,'model_module':MocoModule, 
                    'model_instance':MocoModelInstance,'train':general_training, 'data_dict':general_dataset_dict},
                    
                    'SupCon':{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'train':general_training, 'data_dict':general_dataset_dict},

                    'CEEnsemble': {'params':cross_entropy_ensemble_hparams,'model_module':CrossEntropyEnsembleModule, 
                    'model_instance':CEEnsembleModelInstance, 'train':general_confusion_training, 'data_dict':general_dataset_dict},
                    
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
    
    datasets = ['MNIST','FashionMNIST','KMNIST','CIFAR10', 'CIFAR100']
    ood_datasets = [['FashionMNIST'],['MNIST'],['SVHN'],['SVHN']]
    
    # BASELINES
    # Go through all the models in the current dataset and current OOD dataset
    if base_dict['single_model']== 'Baselines':
        for model_k, model_v in model_dict.items():
            # Checks if model is present in the acceptable single models
            if model_k in acceptable_single_models:
                params = model_dict[model_k]['params']
                train_method = model_dict[model_k]['train']
                model_module = model_dict[model_k]['model_module'] 
                model_instance_method = model_dict[model_k]['model_instance']
                model_data_dict = model_dict[model_k]['data_dict']
                # Try statement to allow the code to continue even if a single run fails
                #train_method(params, model_module, model_instance_method)
                train_method(params, model_module, model_instance_method,model_data_dict)

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
        model_data_dict = model_info['data_dict']
        # Loop through the different datasets and OOD datasets and examine if the model is able to train for the task
        for dataset, ood_dataset in zip(datasets, ood_datasets):
            params['dataset'] = dataset
            params['OOD_dataset'] = ood_dataset
            train_method(params, model_module, model_instance_method, model_data_dict)