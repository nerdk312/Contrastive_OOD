from re import search

from Contrastive_uncertainty.general_hierarchy.callbacks.general_callbacks import  ModelSaving, MMD_distance
from Contrastive_uncertainty.general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD, Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD, Mahalanobis_OOD_Datasets
from Contrastive_uncertainty.general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.general_hierarchy.callbacks.variational_callback import Variational
from Contrastive_uncertainty.general_hierarchy.callbacks.hierarchical_ood import Hierarchical_Mahalanobis
from Contrastive_uncertainty.general_hierarchy.datamodules.datamodule_dict import dataset_dict
from Contrastive_uncertainty.general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection

'''
def callback_dictionary(Datamodule,OOD_Datamodule,config):
    quick_callback = config['quick_callback']
    
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                'Mahalanobis_instance_fine': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance',label_level='fine'),
                'Mahalanobis_fine_fine': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='fine',label_level='fine'),
                'Mahalanobis_coarse_coarse': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='coarse',label_level='coarse'),
                'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
                'Visualisation_instance_fine': Visualisation(Datamodule, OOD_Datamodule,vector_level='instance',label_level='fine',quick_callback=quick_callback),
                'Visualisation_fine_fine': Visualisation(Datamodule, OOD_Datamodule,vector_level='fine',label_level='fine',quick_callback=quick_callback),
                'Visualisation_coarse_coarse': Visualisation(Datamodule, OOD_Datamodule,vector_level='coarse',label_level='coarse',quick_callback=quick_callback),
                'Metrics_instance_fine': MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                'Metrics_fine_fine': MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='fine', label_level='fine', quick_callback=quick_callback),
                'Metrics_coarse_coarse': MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='coarse', label_level='coarse', quick_callback=quick_callback)
                }
    
    return callback_dict
#'IsoForest': IsoForest(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
#'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),
#  'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters,quick_callback=quick_callback),'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
# 'Centroid': Centroid_distance(Datamodule, config['quick_callback']), 'SupCon': SupConLoss(Datamodule, config['quick_callback'])}
'''
# Run name which includes the branch weights
def train_run_name(model_name, config, group=None):
    run_name = "Train_" + model_name + "_DS:"+str(config["dataset"]) +"_Epochs:" + str(config["epochs"]) + "_seed:" +str(config["seed"]) + f'_instance:{config["branch_weights"][0]:.2f}_fine:{config["branch_weights"][1]:.2f}_coarse:{config["branch_weights"][2]:.2f}' 
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
                    'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
                    'Metrics_instance_fine':MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                    'Visualisation_instance_fine': Visualisation(Datamodule, vector_level='instance',label_level='fine',quick_callback=quick_callback),
                    'Variational':Variational(Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback)}
    
    # Automatically adding callbacks for the Mahalanobis distance for each different vector level as well as each different OOD dataset
    Collated_OOD_datamodules = []
    for ood_dataset in config['OOD_dataset']:
        OOD_Datamodule = Datamodule_selection(dataset_dict, ood_dataset, config)
        OOD_callback = {f'Mahalanobis_instance_fine_{ood_dataset}':Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                f'Aggregated {ood_dataset}': Aggregated_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Differing {ood_dataset}': Differing_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Hierarchical {ood_dataset}':Hierarchical_Mahalanobis(Datamodule, OOD_Datamodule,quick_callback=quick_callback)}
        
        callback_dict.update(OOD_callback)
        Collated_OOD_datamodules.append(OOD_Datamodule)
    
    callback_dict.update({'OOD_Dataset_distances': Mahalanobis_OOD_Datasets(Datamodule, Collated_OOD_datamodules, quick_callback=quick_callback)})

    return callback_dict

def specific_callbacks(callback_dict, names):
    desired_callbacks = []
    # Obtain all the different callback keys
    callback_keys = callback_dict.keys()
    
    # Iterate through all the different names which I specify
    for index, name in enumerate(names):
        for key in callback_keys:  # Goes through all the different keys
            if search(name, key):  # Checks if name is part of the substring of key 
                desired_callbacks.append(callback_dict[key]) # Add the specific callback
    
    return desired_callbacks

