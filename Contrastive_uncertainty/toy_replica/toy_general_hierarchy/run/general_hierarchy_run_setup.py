from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.general_callbacks import  ModelSaving,MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.ood_callbacks import Mahalanobis_OOD 
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general_hierarchy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.toy_replica.toy_general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection 

def callback_dictionary(Datamodule,OOD_Datamodule,config):
    val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    num_classes = Datamodule.num_classes
    quick_callback = config['quick_callback']
    inference_clusters = [num_classes]
    

    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Toy_Models'),
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_loader,evaltypes,config['quick_callback']),
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters, quick_callback=quick_callback),
                'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
                'Visualisation': Visualisation(Datamodule, OOD_Datamodule, config['quick_callback']),
                
                }
    
    return callback_dict

