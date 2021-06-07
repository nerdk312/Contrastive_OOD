from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD, Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.toy_replica.toy_general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection, specific_callbacks
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.datamodules.datamodule_dict import dataset_dict
from re import search

# Run name which includes the branch weights
def train_run_name(model_name, config, group=None):
    run_name = "Train_" + model_name + "_DS:"+str(config["dataset"]) +"_Epochs:" + str(config["epochs"]) + "_seed:" +str(config["seed"]) + f'_instance:{config["branch_weights"][0]}_fine:{config["branch_weights"][1]}_coarse:{config["branch_weights"][2]}' 
    if group is not None:
        run_name = group + '_' + run_name
    return run_name


# Generates the callbacks
def callback_dictionary(Datamodule,config):
    #val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    #num_classes = Datamodule.num_classes
    
    quick_callback = config['quick_callback']
    
    # Manually added callbacks
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                    'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback)}
    
    # Iterate through the different vector and label levels to get different metrics and visualisations
    for (vector_level, label_level) in zip(config['vector_level'],config['label_level']):
        additional_callbacks = {f'Metrics_{vector_level}_{label_level}':MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback),
                                f'Visualisation_{vector_level}_{label_level}': Visualisation(Datamodule, vector_level=vector_level,label_level=label_level,quick_callback=quick_callback)}
        callback_dict.update(additional_callbacks)


        # Automatically adding callbacks for the Mahalanobis distance for each different vector level as well as each different OOD dataset
        for ood_dataset in config['OOD_dataset']:
            OOD_Datamodule = Datamodule_selection(dataset_dict,ood_dataset,config)
            OOD_callback = {f'Mahalanobis_{vector_level}_{label_level}_{ood_dataset}':Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level=vector_level, label_level=label_level),
                    f'Aggregated {ood_dataset}': Aggregated_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                    f'Differing {ood_dataset}': Differing_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}
            callback_dict.update(OOD_callback)
    
    return callback_dict



