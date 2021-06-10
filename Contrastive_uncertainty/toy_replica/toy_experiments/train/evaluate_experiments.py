import wandb

# Import parameters for different training methods
from Contrastive_uncertainty.toy_replica.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.toy_replica.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.toy_replica.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.config.hsup_con_bu_params import hsup_con_bu_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.config.hsup_con_td_params import hsup_con_td_hparams
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.config.hsup_con_bu_centroid_params import hsup_con_bu_centroid_hparams


# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.toy_replica.cross_entropy.models.cross_entropy_module import CrossEntropyToy
from Contrastive_uncertainty.toy_replica.moco.models.moco_module import MocoToy
from Contrastive_uncertainty.toy_replica.sup_con.models.sup_con_module import SupConToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_module import HSupConBUToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.models.hsup_con_td_module import HSupConTDToy
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroidToy


# Model instances for the different methods
from Contrastive_uncertainty.toy_replica.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.toy_replica.moco.models.moco_model_instance import ModelInstance as MocoModelInstance
from Contrastive_uncertainty.toy_replica.sup_con.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu.models.hsup_con_bu_model_instance import ModelInstance as HSupConBUModelInstance
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_td.models.hsup_con_td_model_instance import ModelInstance as HSupConTDModelInstance
from Contrastive_uncertainty.toy_replica.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance as HSupConBUCentroidModelInstance


# Import training methods 
from Contrastive_uncertainty.toy_replica.toy_general.train.evaluate_general import evaluation as general_evaluation
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.train.evaluate_general_hierarchy import evaluation as general_hierarchy_evaluation

def evaluate(run_paths,update_dict):    
    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon',
    'PCL','MultiPCL','UnSupConMemory','HSupCon','HSupConBU','HSupConBUCentroid','HSupConTD']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyToy, 
                    'model_instance':CEModelInstance, 'evaluate':general_evaluation},

                    'Moco':{'params':moco_hparams,'model_module':MocoToy, 
                    'model_instance':MocoModelInstance, 'evaluate':general_evaluation},

                    'SupCon':{'params':sup_con_hparams,'model_module':SupConToy, 
                    'model_instance':SupConModelInstance, 'evaluate':general_evaluation},
    }
    
    
    '''
    model_dict = {'HSupConBUCentroid':{'params':hsup_con_bu_centroid_hparams,'model_module': HSupConBUCentroidToy, 
                    'model_instance':HSupConBUCentroidModelInstance, 'evaluate':general_hierarchy_evaluation},
                    
                    'HSupConBU':{'params':hsup_con_bu_hparams,'model_module': HSupConBUToy, 
                    'model_instance':HSupConBUModelInstance,'evaluate':general_hierarchy_evaluation},

                    'HSupConTD':{'params':hsup_con_td_hparams,'model_module': HSupConTDToy, 
                    'model_instance':HSupConTDModelInstance,'evaluate':general_hierarchy_evaluation},
    }
    '''

    # Iterate through the run paths
    for run_path in run_paths:
        api = wandb.Api()    
        # Obtain previous information such as the model type to be able to choose appropriate methods
        previous_run = api.run(path=run_path)
        previous_config = previous_run.config
        model_type = previous_config['model_type']
        # Choosing appropriate methods to resume the training        
        evaluate_method = model_dict[model_type]['evaluate']
        model_module = model_dict[model_type]['model_module'] 
        model_instance_method = model_dict[model_type]['model_instance']
        evaluate_method(run_path, update_dict, model_module, model_instance_method)