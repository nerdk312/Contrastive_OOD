from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD, Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.toy_replica.toy_general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection, specific_callbacks 
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.datamodules.datamodule_dict import dataset_dict
from re import search

# Generates the callbacks
def callback_dictionary(Datamodule,config):
    #val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    #num_classes = Datamodule.num_classes
    ood_dataset = config['OOD_dataset'][0]
    OOD_Datamodule = Datamodule_selection(dataset_dict,ood_dataset,config)
    quick_callback = config['quick_callback']
    #import ipdb; ipdb.set_trace()
    # Manually added callbacks
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                    'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
                    'Aggregated':Aggregated_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                    'Differing':Differing_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}
    
    # Iterate through the different vector and label levels to get different metrics and visualisations
    for (vector_level,label_level) in zip(config['vector_level'],config['label_level']):
        additional_callbacks = {f'Metrics_{vector_level}_{label_level}':MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level=vector_level, label_level=label_level, quick_callback=quick_callback),
                                f'Visualisation_{vector_level}_{label_level}': Visualisation(Datamodule, vector_level=vector_level,label_level=label_level,quick_callback=quick_callback)}
        callback_dict.update(additional_callbacks)


        # Automatically adding callbacks for the Mahalanobis distance for each different vector level as well as each different OOD dataset
        for ood_dataset in config['OOD_dataset']:
            OOD_Datamodule = Datamodule_selection(dataset_dict,ood_dataset,config)
            OOD_callback = {f'Mahalanobis_{vector_level}_{label_level}_{ood_dataset}':Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level=vector_level, label_level=label_level)}
            callback_dict.update(OOD_callback)
    
    return callback_dict


def specific_callbacks(callback_dict, names):
    desired_callbacks = []
    # Obtain all the different callback keys
    callback_keys = callback_dict.keys()
    
    # Iterate through all the different names which I specify
    for index, name in enumerate(names):
        for key in callback_keys: # Goes through all the different keys
            if search(name, key): # Checks if name is part of the substring of key 
                desired_callbacks.append(callback_dict[key]) # Add the specific callback
    
    #for index, name in enumerate(config['callbacks']):
        # Need to obtain list of names which contain the substring of interest
        #import ipdb; ipdb.set_trace()
        #desired_callbacks.append(callback_dict[name])
    #import ipdb; ipdb.set_trace()
    return desired_callbacks


