# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.Contrastive.config.contrastive_params import contrastive_hparams
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams


# Import training methods 
from Contrastive_uncertainty.cross_entropy.train.train_cross_entropy import train as CE_training
from Contrastive_uncertainty.Contrastive.train.train_contrastive import train as Moco_training
from Contrastive_uncertainty.sup_con.train.train_sup_con import train as SupCon_training
from Contrastive_uncertainty.PCL.train.train_pcl import training as PCL_training
from Contrastive_uncertainty.unsup_con_memory.train.train_unsup_con_memory import train as UnSupConMemory_training

def train(base_dict):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon','PCL','UnSupConMemory']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'train':CE_training},                
                    'Moco':{'params':contrastive_hparams,'train':Moco_training},
                    'SupCon':{'params':sup_con_hparams,'train':SupCon_training},
                    'PCL':{'params':pcl_hparams,'train':PCL_training},                
                    'UnSupConMemory':{'params':unsup_con_memory_hparams,'train':UnSupConMemory_training}
                    }
    '''
    model_dict = {
                    'PCL':{'params':pcl_hparams,'train':PCL_training},                
                    'UnSupConMemory':{'params':unsup_con_memory_hparams,'train':UnSupConMemory_training}
                    }
    '''             
    
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

    datasets = ['FashionMNIST','MNIST','KMNIST','CIFAR10']
    ood_datasets = ['MNIST','FashionMNIST','MNIST','SVHN']

    # BASELINES
    # Go through all the models in the current dataset and current OOD dataset
    if base_dict['single_model']== 'Baselines':
        for model_k, model_v in model_dict.items():
            params = model_dict[model_k]['params']
            train_method = model_dict[model_k]['train']
            # Try statement to allow the code to continue even if a single run fails
            train_method(params)
            '''
            try:
                train_method(params)
            except:
                print('f{model_k} training did not work')
            ''' 
    ## SINGLE MODEL
    # Go through a single model on all different datasets
    else:
        # Name of the chosen model
        chosen_model = base_dict['single_model']
        # Specific model dictionary chosen
        model_info = model_dict[chosen_model]
        train_method = model_info['train']
        params = model_info['params']
        # Loop through the different datasets and OOD datasets and examine if the model is able to train for the task
        for dataset, ood_dataset in zip(datasets,ood_datasets):
            params['dataset'] = dataset
            params['OOD_dataset'] = ood_dataset
            train_method(params)
            
            '''
            try:
                train_method(params)
            except:
                print('f{model_k} training did not work when using dataset {dataset} and ood dataset {ood_dataset}')
            '''
