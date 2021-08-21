# Import parameters for different training methods
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams
from Contrastive_uncertainty.sup_con_memory.config.sup_con_memory_params import sup_con_memory_hparams
from Contrastive_uncertainty.moco_margin.config.moco_margin_params import moco_margin_hparams

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
from Contrastive_uncertainty.sup_con_memory.models.sup_con_memory_module import SupConMemoryModule
from Contrastive_uncertainty.moco_margin.models.moco_margin_module import MocoMarginModule


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
from Contrastive_uncertainty.sup_con_memory.models.sup_con_memory_model_instance import ModelInstance as SupConMemoryModelInstance
from Contrastive_uncertainty.moco_margin.models.moco_margin_model_instance import ModelInstance as MocoMarginModelInstance

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

# Import datamodule info
from Contrastive_uncertainty.general.datamodules.datamodule_dict import dataset_dict as general_dataset_dict


# Import training methods 
from Contrastive_uncertainty.general.train.train_general import train as general_training
from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train as general_clustering_training
from Contrastive_uncertainty.general_hierarchy.train.train_general_hierarchy import train as general_hierarchy_training
from Contrastive_uncertainty.general.train.train_general_confusion import train as general_confusion_training

def batch_margin_train(base_dict):    
    acceptable_single_models = ['Baselines',
    #'CE',
    #'Moco',
    #'SupCon'
    #'SupConMemory',
    'MocoMargin',
    # 'PCL',
    # 'MultiPCL',
    # 'UnSupConMemory',
    # 'HSupCon',
    #'HSupConBU',
    # 'HSupConBUCentroid',
    #'HSupConTD',
    # 'VAE',
    # 'CEVAE',
    # 'MocoVAE',
    # 'SupConVAE',
    #'CEEnsemble'
    ]

    # Dict for the model name, parameters and specific training loop
    
    
    model_dict = {'CE':{'params':cross_entropy_hparams,'model_module':CrossEntropyModule,
                    'model_instance':CEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},
        
                    'Moco':{'params':moco_hparams,'model_module':MocoModule, 
                    'model_instance':MocoModelInstance,'train':general_training, 'data_dict':general_dataset_dict},
                    
                    'SupCon':{'params':sup_con_hparams,'model_module':SupConModule, 
                    'model_instance':SupConModelInstance,'train':general_training, 'data_dict':general_dataset_dict},

                    'SupConMemory':{'params':sup_con_memory_hparams,'model_module':SupConMemoryModule, 
                    'model_instance':SupConMemoryModelInstance, 'train':general_training, 'data_dict':general_dataset_dict},

                    'MocoMargin':{'params':moco_margin_hparams,'model_module':MocoMarginModule, 
                    'model_instance':MocoMarginModelInstance, 'train':general_training, 'data_dict':general_dataset_dict},

                    'HSupConBUCentroid':{'params':hsup_con_bu_centroid_hparams,'model_module':HSupConBUCentroidModule, 
                    'model_instance':HSupConBUCentroidModelInstance, 'train':general_hierarchy_training, 'data_dict':general_dataset_dict},
                    
                    'HSupConBU':{'params':hsup_con_bu_hparams,'model_module':HSupConBUModule, 
                    'model_instance':HSupConBUModelInstance,'train':general_hierarchy_training, 'data_dict':general_dataset_dict},

                    'HSupConTD':{'params':hsup_con_td_hparams,'model_module':HSupConTDModule, 
                    'model_instance':HSupConTDModelInstance,'train':general_hierarchy_training, 'data_dict':general_dataset_dict},

                    'CEVAE':{'params':cross_entropy_vae_hparams,'model_module':CrossEntropyVAEModule,
                    'model_instance':CrossEntropyVAEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},

                    'MocoVAE':{'params':moco_vae_hparams,'model_module':MocoVAEModule,
                    'model_instance':MocoVAEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},

                    'SupConVAE':{'params':sup_con_vae_hparams,'model_module':SupConVAEModule,
                    'model_instance':SupConVAEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},

                    'VAE':{'params':vae_hparams,'model_module':VAEModule,
                    'model_instance':VAEModelInstance,'train':general_training, 'data_dict':general_dataset_dict},
                    
                    'CEEnsemble': {'params':cross_entropy_ensemble_hparams,'model_module':CrossEntropyEnsembleModule, 
                    'model_instance':CEEnsembleModelInstance, 'train':general_confusion_training, 'data_dict':general_dataset_dict},
                    
    }
    
    

    # Update the seed
    margins = [1.5,2.0,2.5,3.0,3.5]
    #seeds = [26,25,50,75,100] # Additional seed for the case of supcon memory run
    for i in range(len(margins)):
        # Update the seed
        base_dict['margin'] = margins[i]

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