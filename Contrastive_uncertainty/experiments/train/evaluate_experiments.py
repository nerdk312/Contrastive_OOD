import wandb

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
from Contrastive_uncertainty.ensemble.config.cross_entropy_ensemble_params import cross_entropy_ensemble_hparams


from Contrastive_uncertainty.vae_models.vae.config.vae_params import vae_hparams
from Contrastive_uncertainty.vae_models.cross_entropy_vae.config.cross_entropy_vae_params import cross_entropy_vae_hparams
from Contrastive_uncertainty.vae_models.sup_con_vae.config.sup_con_vae_params import sup_con_vae_hparams
from Contrastive_uncertainty.vae_models.moco_vae.config.moco_vae_params import moco_vae_hparams



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
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_module import CrossEntropyEnsembleModule


from Contrastive_uncertainty.vae_models.vae.models.vae_module import VAEModule
from Contrastive_uncertainty.vae_models.cross_entropy_vae.models.cross_entropy_vae_module import CrossEntropyVAEModule
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_module import SupConVAEModule
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_module import MocoVAEModule


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
from Contrastive_uncertainty.ensemble.models.cross_entropy_ensemble_model_instance import ModelInstance as CEEnsembleModelInstance


from Contrastive_uncertainty.vae_models.vae.models.vae_model_instance import ModelInstance as VAEModelInstance
from Contrastive_uncertainty.vae_models.cross_entropy_vae.models.cross_entropy_vae_model_instance import ModelInstance as CrossEntropyVAEModelInstance
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_model_instance import ModelInstance as SupConVAEModelInstance
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_model_instance import ModelInstance as MocoVAEModelInstance


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

                    'HSupConBUCentroid':{'params':hsup_con_bu_centroid_hparams,'model_module':HSupConBUCentroidModule, 
                    'model_instance':HSupConBUCentroidModelInstance, 'evaluate':general_hierarchy_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},
                    
                    'HSupConBU':{'params':hsup_con_bu_hparams,'model_module':HSupConBUModule, 
                    'model_instance':HSupConBUModelInstance,'evaluate':general_hierarchy_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'HSupConTD':{'params':hsup_con_td_hparams,'model_module':HSupConTDModule, 
                    'model_instance':HSupConTDModelInstance,'evaluate':general_hierarchy_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},
    
                    'CEVAE':{'params':cross_entropy_vae_hparams,'model_module':CrossEntropyVAEModule,
                    'model_instance':CrossEntropyVAEModelInstance,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'MocoVAE':{'params':moco_vae_hparams,'model_module':MocoVAEModule,
                    'model_instance':MocoVAEModelInstance,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'SupConVAE':{'params':sup_con_vae_hparams,'model_module':SupConVAEModule,
                    'model_instance':SupConVAEModelInstance,'evaluate':general_evaluation,
                    'data_dict':general_dataset_dict, 'ood_dict':general_OOD_dict},

                    'VAE':{'params':vae_hparams,'model_module':VAEModule,
                    'model_instance':VAEModelInstance,'evaluate':general_evaluation,
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
