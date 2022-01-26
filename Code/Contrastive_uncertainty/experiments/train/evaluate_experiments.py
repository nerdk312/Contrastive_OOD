import wandb

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



# Import evaluate
from Contrastive_uncertainty.general.train.evaluate_general import evaluation as general_evaluation
from Contrastive_uncertainty.general_clustering.train.evaluate_general_clustering import evaluation as general_clustering_evaluation
from Contrastive_uncertainty.general_hierarchy.train.evaluate_general_hierarchy import evaluation as general_hierarchy_evaluation
from Contrastive_uncertainty.general.train.evaluate_general_confusion import evaluation as general_confusion_evaluation


from Contrastive_uncertainty.general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict, OOD_dict as general_OOD_dict
#from Contrastive_uncertainty.general_hierarchy.datamodules.datamodule_dict import dataset_dict as general_hierarchy_dataset_dict, OOD_dict as general_hierarchy_OOD_dict



def evaluate(run_paths,update_dict):    
    
    # Dict for the model name, parameters and specific training loop
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyModule, 
                    'model_instance':CEModelInstance, 'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'Moco':{'params':moco_hparams,'model_module':MocoModule, 
                    'model_instance':MocoModelInstance, 'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'SupCon':{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance, 'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'CEEnsemble': {'params':cross_entropy_ensemble_hparams,'model_module':CrossEntropyEnsembleModule, 
                    'model_instance':CEEnsembleModelInstance, 'evaluate':general_confusion_evaluation, 
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},
    }   
    

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
        model_data_dict = model_dict[model_type]['data_dict']
        model_ood_dict = model_dict[model_type]['ood_dict']
        evaluate_method(run_path, update_dict, model_module, model_instance_method, model_data_dict,model_ood_dict)
