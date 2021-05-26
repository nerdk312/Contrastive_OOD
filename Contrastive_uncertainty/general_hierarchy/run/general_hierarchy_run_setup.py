from Contrastive_uncertainty.general_hierarchy.callbacks.general_callbacks import  ModelSaving, MMD_distance
from Contrastive_uncertainty.general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD 
from Contrastive_uncertainty.general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes

from Contrastive_uncertainty.general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection 

def callback_dictionary(Datamodule,OOD_Datamodule,config):
    quick_callback = config['quick_callback']
    
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                'Mahalanobis_instance_fine': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance',label_level='fine'),
                'Mahalanobis_instance_coarse': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance',label_level='coarse'),
                'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
                'Visualisation_instance_fine': Visualisation(Datamodule, OOD_Datamodule,vector_level='instance',label_level='fine',quick_callback=quick_callback),
                'Visualisation_instance_coarse': Visualisation(Datamodule, OOD_Datamodule,vector_level='instance',label_level='coarse',quick_callback=quick_callback),
                'Metrics_instance_fine': MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                'Metrics_instance_coarse': MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='coarse', quick_callback=quick_callback)
                }
    
    return callback_dict
#'IsoForest': IsoForest(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
#'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),
#  'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters,quick_callback=quick_callback),'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
# 'Centroid': Centroid_distance(Datamodule, config['quick_callback']), 'SupCon': SupConLoss(Datamodule, config['quick_callback'])}