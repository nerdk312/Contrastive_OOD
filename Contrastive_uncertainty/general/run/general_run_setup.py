from re import search

from pytorch_lightning.core import datamodule

from Contrastive_uncertainty.general.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, Mahalanobis_OOD_Datasets, Mahalanobis_OvO, Mahalanobis_OvR, Class_Mahalanobis_OOD, Data_Augmented_Mahalanobis  #Euclidean_OOD, IsoForest
from Contrastive_uncertainty.general.callbacks.experimental_ood_callbacks import Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD
from Contrastive_uncertainty.general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general.callbacks.typicality_ood_callback import Typicality_OVR, Typicality_OVO, Typicality_OVR_diff_bsz, Typicality_General_Point, Typicality_General_Point_updated, Typicality_OVR_diff_batch_updated
from Contrastive_uncertainty.general.callbacks.marginal_typicality_ood_callback import Marginal_Typicality_OOD_detection, Marginal_Typicality_entropy_mean


from Contrastive_uncertainty.general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.general.callbacks.variational_callback import Variational
from Contrastive_uncertainty.general.callbacks.relative_mahalanobis_callback import Relative_Mahalanobis, Class_Relative_Mahalanobis, Class_Inverted_Relative_Mahalanobis
from Contrastive_uncertainty.general.callbacks.one_dim_mahalanobis_callback import One_Dim_Mahalanobis,One_Dim_Relative_Mahalanobis, One_Dim_Shared_Mahalanobis, One_Dim_Shared_Relative_Mahalanobis, Class_One_Dim_Mahalanobis, \
    Class_One_Dim_Relative_Mahalanobis, One_Dim_Background_Mahalanobis
    
from Contrastive_uncertainty.general.callbacks.one_dim_mahalanobis_variance_callback import One_Dim_Mahalanobis_Variance, One_Dim_Relative_Mahalanobis_Variance, Class_One_Dim_Relative_Mahalanobis_Variance
from Contrastive_uncertainty.general.callbacks.one_dim_mahalanobis_similarity_callback import One_Dim_Mahalanobis_Similarity, Class_One_Dim_Mahalanobis_OOD_Similarity

from Contrastive_uncertainty.general.callbacks.oracle_hierarchical_ood import Oracle_Hierarchical_Metrics, Hierarchical_Random_Coarse, Hierarchical_Subclusters_OOD
from Contrastive_uncertainty.general.callbacks.isolation_forest_callback import IForest
from Contrastive_uncertainty.general.callbacks.analysis_callback import Dataset_class_variance, Dataset_class_radii, Centroid_distances, Centroid_relative_distances, Class_Radii_histograms
from Contrastive_uncertainty.general.callbacks.total_centroid_similarity_callback import Total_Centroid_KL, Class_Centroid_Radii_Overlap

from Contrastive_uncertainty.general.callbacks.confusion_log_probability_callback import ConfusionLogProbability
from Contrastive_uncertainty.general.callbacks.bottom_k_mahalanobis_callback import Bottom_K_Mahalanobis, Bottom_K_Mahalanobis_Difference
from Contrastive_uncertainty.general.callbacks.feature_entropy_callback import Feature_Entropy
from Contrastive_uncertainty.general.callbacks.one_dim_typicality_callback import One_Dim_Typicality, One_Dim_Typicality_Class, One_Dim_Typicality_Marginal_Oracle,\
    One_Dim_Typicality_Marginal, One_Dim_Typicality_Normalised_Marginal,\
    Point_One_Dim_Class_Typicality_Normalised, Point_One_Dim_Relative_Class_Typicality_Analysis, Point_One_Dim_Relative_Class_Typicality_Normalised,\
    Data_Augmented_Point_One_Dim_Class_Typicality_Normalised, Alternative_Data_Augmented_Point_One_Dim_Class_Typicality_Normalised

from Contrastive_uncertainty.general.callbacks.one_dim_typicality_analysis_callback import Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis, Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Single_Variance_Analysis

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
    typicality_batch = config['typicality_batch']
    typicality_bootstrap = config['typicality_bootstrap']


    
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
                #  # Callbacks related to typicality as well as OVR and OVO classification
                f'Typicality_OVR_{ood_dataset}': Typicality_OVR(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                f'Typicality_OVO_{ood_dataset}': Typicality_OVO(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                #f'Typicality_OVR_diff_bsz_{ood_dataset}': Typicality_OVR_diff_bsz(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                #f'Typicality_OVR_diff_bsz_updated_{ood_dataset}': Typicality_OVR_diff_batch_updated(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                
                f'Typicality General Point {ood_dataset}': Typicality_General_Point(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                f'Typicality General Point Updated {ood_dataset}': Typicality_General_Point_updated(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),

                f'Marginal Typicality OOD {ood_dataset}': Marginal_Typicality_OOD_detection(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,typicality_bsz=typicality_batch),
                f'Marginal Typicality Entropy Mean OOD {ood_dataset}': Marginal_Typicality_entropy_mean(Datamodule,OOD_Datamodule, quick_callback=quick_callback,typicality_bsz=typicality_batch),

                f'One Dim Typicality {ood_dataset}':One_Dim_Typicality(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'One Dim Typicality Class {ood_dataset}':One_Dim_Typicality_Class(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                
                f'One Dim Typicality Marginal Oracle {ood_dataset}': One_Dim_Typicality_Marginal_Oracle(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'One Dim Typicality Marginal Batch {ood_dataset}': One_Dim_Typicality_Marginal(Datamodule,OOD_Datamodule,quick_callback=quick_callback,typicality_bsz=config['typicality_batch']),
                f'One Dim Typicality Normalised Marginal Batch {ood_dataset}': One_Dim_Typicality_Normalised_Marginal(Datamodule,OOD_Datamodule,quick_callback=quick_callback,typicality_bsz=config['typicality_batch']),
                f'Point One Dim Class Typicality Normalised {ood_dataset}':Point_One_Dim_Class_Typicality_Normalised(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Point One Dim Relative Class Typicality Normalised {ood_dataset}': Point_One_Dim_Relative_Class_Typicality_Normalised(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Data Augmented Point One Dim Class Typicality Normalised {ood_dataset}':Data_Augmented_Point_One_Dim_Class_Typicality_Normalised(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Data Augmented Alternative Point One Dim Class Typicality Normalised {ood_dataset}':Alternative_Data_Augmented_Point_One_Dim_Class_Typicality_Normalised(Datamodule,OOD_Datamodule, quick_callback),
                f'Data Augmented Mahalanobis {ood_dataset}': Data_Augmented_Mahalanobis(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                f'Point One Dim Relative Class Typicality Analysis {ood_dataset}': Point_One_Dim_Relative_Class_Typicality_Analysis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Data Augmented Point One Dim Marginal Typicality Normalised Variance Analysis {ood_dataset}':Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Variance_Analysis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Data Augmented Point One_Dim Marginal Typicality Normalised Single Variance Analysis':Data_Augmented_Point_One_Dim_Marginal_Typicality_Normalised_Single_Variance_Analysis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),

                

                
                f'IForest {ood_dataset}': IForest(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Class Variance': Dataset_class_variance(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Class Radii': Dataset_class_radii(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                
                f'Centroid Distances': Centroid_distances(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Centroid Relative Distances': Centroid_relative_distances(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Class Radii Histograms {ood_dataset}': Class_Radii_histograms(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine'),
                f'Total Centroid KL': Total_Centroid_KL(Datamodule, quick_callback=quick_callback),
                f'Class Centroid Radii Overlap': Class_Centroid_Radii_Overlap(Datamodule, quick_callback=quick_callback),

                f'OVR classification {ood_dataset}':Mahalanobis_OvR(Datamodule, OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                f'OVO classification {ood_dataset}':Mahalanobis_OvO(Datamodule, OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback),

                # One dimensional mahalanobis callbacks
                f'One Dimensional Mahalanobis {ood_dataset}': One_Dim_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'One Dimensional Shared Mahalanobis {ood_dataset}': One_Dim_Shared_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'One Dimensional Relative Mahalanobis {ood_dataset}': One_Dim_Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'One Dimensional Shared Relative Mahalanobis {ood_dataset}': One_Dim_Shared_Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'One Dimensional Background Mahalanobis {ood_dataset}': One_Dim_Background_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                
                # One dimensional mahalanobis variance callbacks
                f'One Dimensional Mahalanobis Variance {ood_dataset}': One_Dim_Mahalanobis_Variance(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'One Dimensional Relative Mahalanobis Variance {ood_dataset}': One_Dim_Relative_Mahalanobis_Variance(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Class One Dimensional Relative Mahalanobis Variance {ood_dataset}': Class_One_Dim_Relative_Mahalanobis_Variance(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                
                # Class specific one dimensional mahalanobis callbacks
                f'Class One Dimensional Mahalanobis {ood_dataset}': Class_One_Dim_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Class One Dimensional Relative Mahalanobis {ood_dataset}': Class_One_Dim_Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),

                f'One Dimensional Mahalanobis Similarity': One_Dim_Mahalanobis_Similarity(Datamodule, quick_callback=quick_callback),
                f'Class One Dimensional Mahalanobis OOD Similarity {ood_dataset}': Class_One_Dim_Mahalanobis_OOD_Similarity(Datamodule,OOD_Datamodule, quick_callback=quick_callback),

                f'Bottom K Mahalanobis OOD {ood_dataset}': Bottom_K_Mahalanobis(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,k_values = 3),
                f'Bottom K Mahalanobis Difference OOD {ood_dataset}': Bottom_K_Mahalanobis_Difference(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,k_values = 3),
                
                f'Feature Entropy': Feature_Entropy(Datamodule,OOD_Datamodule,quick_callback),
                f'Confusion Log Probability': ConfusionLogProbability(Datamodule,quick_callback),

                f'Relative Mahalanobis {ood_dataset}': Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Mahalanobis Distance {ood_dataset}': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                
                f'Class Mahalanobis {ood_dataset}': Class_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                f'Class Relative Mahalanobis {ood_dataset}': Class_Relative_Mahalanobis(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                f'Class Inverted Relative Mahalanobis {ood_dataset}': Class_Inverted_Relative_Mahalanobis(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance', label_level='fine'),
                
                f'Oracle Hierarchical Metrics {ood_dataset}':Oracle_Hierarchical_Metrics(Datamodule, OOD_Datamodule,quick_callback=quick_callback),
                f'Hierarchical_Random_Coarse {ood_dataset}' : Hierarchical_Random_Coarse(Datamodule, OOD_Datamodule,quick_callback=quick_callback),
                f'Hierarchical Subclusters 10 {ood_dataset}' : Hierarchical_Subclusters_OOD(Datamodule, OOD_Datamodule,quick_callback=quick_callback, vector_level='fine',label_level='fine',num_clusters=10)}
                
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
        for key in callback_keys: # Goes through all the different keys
            if search(name, key): # Checks if name is part of the substring of key 
                desired_callbacks.append(callback_dict[key]) # Add the specific callback
    
    return desired_callbacks


#'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
#'Variational':Variational(Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback)
#f'Aggregated {ood_dataset}': Aggregated_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
#f'Differing {ood_dataset}': Differing_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),