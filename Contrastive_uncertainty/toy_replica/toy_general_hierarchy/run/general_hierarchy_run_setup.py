from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD 
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.toy_replica.toy_general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection 

def callback_dictionary(Datamodule,OOD_Datamodule,config):
    #val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    #num_classes = Datamodule.num_classes
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

