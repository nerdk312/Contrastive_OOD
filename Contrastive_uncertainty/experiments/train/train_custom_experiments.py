# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams
from Contrastive_uncertainty.hierarchical_models.HSupCon.config.hsup_con_params import hsup_con_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConBU.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.hierarchical_models.HSupConTD.config.hsup_con_td_params import hsup_con_td_hparams
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.config.hsup_con_bu_centroid_params import hsup_con_bu_centroid_hparams
from Contrastive_uncertainty.multi_PCL.config.multi_pcl_params import multi_pcl_hparams
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams

# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.sup_con.models.sup_con_module import SupConModule
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_module import HSupConModule
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_module import HSupConBUModule
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_module import HSupConTDModule
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroidModule
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule

# Model instances for the different methods
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.sup_con.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance as PCLModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_model_instance import ModelInstance as HSupConModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_model_instance import ModelInstance as HSupConBUModelInstance
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance as HSupConBUCentroidModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_model_instance import ModelInstance as HSupConTDModelInstance
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance as MultiPCLModelInstance
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance as UnSupConMemoryModelInstance


# Import training methods 
from Contrastive_uncertainty.general.train.train_general import train as general_training
#from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train as general_clustering_training
from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train as general_hierarchy_training

def train(base_dict):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon',
    'PCL','MultiPCL','UnSupConMemory','HSupCon','HSupConBU','HSupConTD']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyModule,
                    'model_instance':CEModelInstance,'train':general_training},
        
                    'Moco':{'params':moco_hparams,'model_module':MocoModule, 
                    'model_instance':MocoModelInstance,'train':general_training},
                    
                    'SupCon':{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'train':general_training},
                    
                    'HSupConBUCentroid':{'params':hsup_con_bu_centroid_hparams,'model_module':HSupConBUCentroidModule, 
                    'model_instance':HSupConBUCentroidModelInstance, 'train':general_hierarchy_training},
                    
                    'HSupConBU':{'params':hsup_con_bu_hparams,'model_module':HSupConBUModule, 
                    'model_instance':HSupConBUModelInstance,'train':general_hierarchy_training},

                    'HSupConTD':{'params':hsup_con_td_hparams,'model_module':HSupConTDModule, 
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
    
    experiment_models = ['HSupConBU',
                        'HSupConBU',
                        'HSupConBU',
                        'HSupConBU',
                        'HSupConBU',
                        'HSupConBU',
                        'HSupConBU']

    script_params_dict = {

                    'branch_weights': [[1., 0., 0.],
                                        [0., 1., 0.],
                                        [0., 0., 1.],
                                        [1./2, 1./2, 0.],
                                        [1./2, 0., 1./2],
                                        [0., 1./2, 1./2],
                                        [1./3, 1./3, 1./3]]
                    
    }

    # Iterates through the model
    for i, chosen_model in enumerate(experiment_models):
        # Selects the model, the train method and the params
        model_info = model_dict[chosen_model]
        train_method = model_info['train']
        params = model_info['params']
        model_module = model_info['model_module'] 
        model_instance_method = model_info['model_instance']
        # Goes through the keys and the values
        for script_key, script_value in script_params_dict.items():
            # Update the params
            if script_key in params:
                params[script_key] = script_value[i]

        # Perform training
        train_method(params, model_module, model_instance_method)
