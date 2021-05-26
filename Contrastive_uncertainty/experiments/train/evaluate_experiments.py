import wandb

# Importing the different lightning modules for the baselines
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.Contrastive.models.contrastive_module import ContrastiveModule
from Contrastive_uncertainty.sup_con.models.sup_con_module import SupConModule
from Contrastive_uncertainty.PCL.models.pcl_module import PCLModule
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_module import HSupConModule
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_module import HSupConBUModule
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_module import HSupConTDModule
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_module import HSupConBUCentroid
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_module import MultiPCLModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule


# Model instances for the different methods
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance as CEModelInstance
from Contrastive_uncertainty.Contrastive.models.contrastive_model_instance import ModelInstance as ContrastiveModelInstance
from Contrastive_uncertainty.sup_con.models.sup_con_model_instance import ModelInstance as SupConModelInstance
from Contrastive_uncertainty.PCL.models.pcl_model_instance import ModelInstance as PCLModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupCon.models.hsup_con_model_instance import ModelInstance as HSupConModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupConBU.models.hsup_con_bu_model_instance import ModelInstance as HSupConBUModelInstance
from Contrastive_uncertainty.hierarchical_models.hsup_con_bu_centroid.models.hsup_con_bu_centroid_model_instance import ModelInstance as HSupConBUCentroidModelInstance
from Contrastive_uncertainty.hierarchical_models.HSupConTD.models.hsup_con_td_model_instance import ModelInstance as HSupConTDModelInstance
from Contrastive_uncertainty.multi_PCL.models.multi_pcl_model_instance import ModelInstance as MultiPCLModelInstance
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance as UnSupConMemoryModelInstance

# Import evaluate
from Contrastive_uncertainty.general.train.evaluate_general import evaluation as general_evaluation
from Contrastive_uncertainty.general_clustering.train.evaluate_general_clustering import evaluation as general_clustering_evaluation
from Contrastive_uncertainty.general_hierarchy.train.evaluate_general_hierarchy import evaluation as general_hierarchy_evaluation


def evaluate(run_paths):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon','PCL','UnSupConMemory','HSupCon']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'model_module':CrossEntropyModule,
                 'model_instance':CEModelInstance,'evaluate':general_evaluation},                
                    
                    'Moco':{'model_module':ContrastiveModule, 
                    'model_instance':ContrastiveModelInstance,'evaluate':general_evaluation},
                    
                    'SupCon':{'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'evaluate':general_evaluation},
                    
                    'PCL':{'model_module':PCLModule,
                    'model_instance':PCLModelInstance,'evaluate':general_clustering_evaluation},

                    'UnSupConMemory':{'model_module':UnSupConMemoryModule,
                    'model_instance':UnSupConMemoryModelInstance,'evaluate':general_clustering_evaluation}
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
        evaluate_method(run_path,model_module,model_instance_method)