# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.Contrastive.config.contrastive_params import contrastive_hparams
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams
from Contrastive_uncertainty.multi_PCL.config.multi_pcl_params import multi_pcl_hparams
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams

# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.Contrastive.models.contrastive_module import ContrastiveModule
from Contrastive_uncertainty.sup_con.models.sup_con_module import SupConModule
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule

# Model instances for the different methods
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.Contrastive.models.contrastive_model_instance import ModelInstance as ContrastiveModelInstance
from Contrastive_uncertainty.sup_con.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance as PCLModelInstance
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance as MultiPCLModelInstance
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance as UnSupConMemoryModelInstance

# Import training methods 
from Contrastive_uncertainty.general.train.train_general import train as general_training
from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train as general_clustering_training

def train(base_dict):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon','PCL','MultiPCL','UnSupConMemory']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyModule,
                 'model_instance':CEModelInstance,'train':general_training},                
                    
                    'Moco':{'params':contrastive_hparams,'model_module':ContrastiveModule, 
                    'model_instance':ContrastiveModelInstance,'train':general_training},
                    
                    'SupCon':{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'train':general_training},
                    
                    'PCL':{'params':pcl_hparams,'model_module':PCLModule,
                    'model_instance':PCLModelInstance,'train':general_clustering_training},

                    'MultiPCL':{'params':multi_pcl_hparams,'model_module':MultiPCLModule,
                    'model_instance':MultiPCLModelInstance,'train':general_clustering_training},

                    'UnSupConMemory':{'params':unsup_con_memory_hparams,'model_module':UnSupConMemoryModule,
                    'model_instance':UnSupConMemoryModelInstance,'train':general_clustering_training}
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

    datasets = ['FashionMNIST','MNIST','KMNIST','CIFAR10']
    ood_datasets = ['MNIST','FashionMNIST','MNIST','SVHN']

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
            
