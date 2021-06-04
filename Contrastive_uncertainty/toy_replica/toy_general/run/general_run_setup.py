from re import search

from Contrastive_uncertainty.toy_replica.toy_general.callbacks.general_callbacks import ModelSaving, MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.ood_callbacks import Mahalanobis_OOD, Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD 
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import dataset_dict


def train_run_name(model_name, config, group=None):
    run_name = 'Train_' + model_name + '_DS:'+str(config['dataset']) +'_Epochs:'+ str(config['epochs']) + '_seed:' +str(config['seed'])  
    if group is not None:
        run_name = group + '_' + run_name
    return run_name

def eval_run_name(model_name,config, group=None):
    run_name = 'Eval_' + model_name + '_DS:'+str(config['dataset']) +'_Epochs:'+ str(config['epochs']) + '_seed:' +str(config['seed'])   
    if group is not None:
        run_name = group + '_' + run_name
    return run_name

def Datamodule_selection(data_dict, dataset, config):
    # Information regarding the configuration of the data module for the specific task
    datamodule_info =  data_dict[dataset] # Specific module
    #import ipdb; ipdb.set_trace()
    Datamodule = datamodule_info['module'](data_dir= './',batch_size = config['bsz'],seed = config['seed'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule


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

def specific_callbacks(callback_dict, names):
    desired_callbacks = []
    # Obtain all the different callback keys
    callback_keys = callback_dict.keys()
    
    # Iterate through all the different names which I specify
    for index, name in enumerate(names):
        for key in callback_keys: # Goes through all the different keys
            if search(name, key): # Checks if name is part of the substring of key 
                desired_callbacks.append(callback_dict[key]) # Add the specific callback
    
    return desired_callbacks

