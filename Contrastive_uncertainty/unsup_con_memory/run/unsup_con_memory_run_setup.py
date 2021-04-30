from Contrastive_uncertainty.general_clustering.run.general_clustering_run_setup import \
    train_run_name,eval_run_name, Datamodule_selection,Channel_selection

from Contrastive_uncertainty.unsup_con_memory.callbacks.general_callbacks import  ModelSaving,SupConLoss,Uniformity,MMD_distance,Centroid_distance
from Contrastive_uncertainty.unsup_con_memory.callbacks.ood_callbacks import  Mahalanobis_OOD, Euclidean_OOD  #,ImagePredictionLogger, OOD_ROC, OOD_confusion_matrix,
from Contrastive_uncertainty.unsup_con_memory.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.unsup_con_memory.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    samples = next(iter(val_loader))
    sample_size = config['bsz']
    num_classes = config['num_classes']
    OOD_val_loader = OOD_Datamodule.val_dataloader()
    
    OOD_samples = next(iter(OOD_val_loader))
    if isinstance(config['num_cluster'], list) or isinstance(config['num_cluster'], tuple):
        num_clusters = config['num_cluster']
    else:  
        num_clusters = [config['num_cluster']]

    callback_dict = {'Model_saving':ModelSaving(config['model_saving']), 
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_loader,evaltypes,config['quick_callback']),
                
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=num_clusters, quick_callback=config['quick_callback']),
                
                'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule, num_inference_clusters=num_clusters, quick_callback=config['quick_callback']),'MMD': MMD_distance(Datamodule, config['quick_callback']),
                
                'Visualisation': Visualisation(Datamodule, OOD_Datamodule,num_classes,config['quick_callback']),'Uniformity': Uniformity(2, Datamodule, config['quick_callback']),
                
                'Centroid': Centroid_distance(Datamodule, config['quick_callback']), 'SupCon': SupConLoss(Datamodule, config['quick_callback'])}
    
    return callback_dict

#'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),