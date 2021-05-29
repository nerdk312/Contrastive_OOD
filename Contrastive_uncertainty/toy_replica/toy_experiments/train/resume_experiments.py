import wandb

from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams

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


# Import resume methods
from Contrastive_uncertainty.general.train.resume_general import resume as general_resume
from Contrastive_uncertainty.general_clustering.train.resume_general_clustering import resume as general_clustering_resume
from Contrastive_uncertainty.general_hierarchy.train.resume_general_hierarchy import resume as general_hierarchy_resume


def resume(run_paths, trainer_dict):    
    acceptable_single_models = ['Baselines','CE','Moco','SupCon','PCL','UnSupConMemory','HSupCon']

    # Dict for the model name, parameters and specific training loop
    
    model_dict = {'CE':{'model_module':CrossEntropyModule,
                 'model_instance':CEModelInstance,'resume':general_resume},                
                    
                    'Moco':{'model_module':ContrastiveModule, 
                    'model_instance':ContrastiveModelInstance,'resume':general_resume},
                    
                    'SupCon':{'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'resume':general_resume},
                    
                    'PCL':{'model_module':PCLModule,
                    'model_instance':PCLModelInstance,'resume':general_clustering_resume},

                    'HSupCon':{'model_module':HSupConModule, 
                    'model_instance':HSupConModelInstance,'resume':general_hierarchy_resume},

                    'UnSupConMemory':{'model_module':UnSupConMemoryModule,
                    'model_instance':UnSupConMemoryModelInstance,'resume':general_clustering_resume}
                    }

    # Iterate through the run paths
    for run_path in run_paths:
        api = wandb.Api()    
        # Obtain previous information such as the model type to be able to choose appropriate methods
        previous_run = api.run(path=run_path)
        previous_config = previous_run.config
        model_type = previous_config['model_type']
        # Choosing appropriate methods to resume the training        
        resume_method = model_dict[model_type]['resume']
        model_module = model_dict[model_type]['model_module'] 
        model_instance_method = model_dict[model_type]['model_instance']
        resume_method(run_path,trainer_dict,model_module,model_instance_method)