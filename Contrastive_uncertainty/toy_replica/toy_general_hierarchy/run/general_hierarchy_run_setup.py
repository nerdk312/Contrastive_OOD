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
    

    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Toy_Models'),
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_loader,evaltypes,config['quick_callback']),
                'Mahalanobis_instance_fine': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance',label_level='fine'),
                'Mahalanobis_instance_coarse': Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level='instance',label_level='coarse'),
                'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
                'Visualisation_instance_fine': Visualisation(Datamodule, OOD_Datamodule,quick_callback=config['quick_callback'],vector_level='instance',label_level='fine'),
                'Visualisation_instance_coarse': Visualisation(Datamodule, OOD_Datamodule,vector_level='instance',quick_callback=config['quick_callback'],label_level='coarse')
                }
    
    return callback_dict

