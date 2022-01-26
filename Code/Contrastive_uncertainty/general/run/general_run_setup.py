from re import search

from pytorch_lightning.core import datamodule

from Contrastive_uncertainty.general.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD
from Contrastive_uncertainty.general.callbacks.visualisation_callback import Visualisation

from Contrastive_uncertainty.general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
    
from Contrastive_uncertainty.general.callbacks.analysis_callback import Dataset_class_variance, Dataset_class_radii, Centroid_distances, Centroid_relative_distances, Class_Radii_histograms
from Contrastive_uncertainty.general.callbacks.total_centroid_similarity_callback import Total_Centroid_KL, Class_Centroid_Radii_Overlap

from Contrastive_uncertainty.general.callbacks.confusion_log_probability_callback import ConfusionLogProbability
from Contrastive_uncertainty.general.callbacks.feature_entropy_callback import Feature_Entropy


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
    num_augmentations = config['num_augmentations']
    datamodule_info =  data_dict[dataset] # Specific module
    Datamodule = datamodule_info['module'](data_dir= './',batch_size = config['bsz'],seed = config['seed'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    Datamodule.multi_transforms = datamodule_info['multi_transform'](num_augmentations)
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule


def callback_dictionary(Datamodule,config,data_dict):
    #val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    #num_classes = Datamodule.num_classes
    
    quick_callback = config['quick_callback']
    
    # Manually added callbacks
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                    'Metrics':MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                    'Visualisation': Visualisation(Datamodule, vector_level='instance',label_level='fine',quick_callback=quick_callback),
                    'MMD': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback)}
                    
    # Automatically adding callbacks for the Mahalanobis distance for each different vector level as well as each different OOD dataset
    # Collated list of OOD datamodules
    Collated_OOD_datamodules = []
    #import ipdb; ipdb.set_trace()
    for ood_dataset in config['OOD_dataset']:
        OOD_Datamodule = Datamodule_selection(data_dict, ood_dataset, config)
        OOD_callback = {                
                # Callbacks related to OOD datasets
                f'Class Variance': Dataset_class_variance(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Class Radii': Dataset_class_radii(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                
                f'Centroid Distances': Centroid_distances(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Centroid Relative Distances': Centroid_relative_distances(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                
                f'Class Radii Histograms {ood_dataset}': Class_Radii_histograms(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Total Centroid KL': Total_Centroid_KL(Datamodule, quick_callback=quick_callback),
                f'Class Centroid Radii Overlap': Class_Centroid_Radii_Overlap(Datamodule, quick_callback=quick_callback),

                f'Feature Entropy': Feature_Entropy(Datamodule,OOD_Datamodule,quick_callback),
                f'Confusion Log Probability': ConfusionLogProbability(Datamodule,quick_callback),

                f'Mahalanobis Distance {ood_dataset}': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine')}
                
        callback_dict.update(OOD_callback)
        Collated_OOD_datamodules.append(OOD_Datamodule)
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