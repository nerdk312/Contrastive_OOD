from Contrastive_uncertainty.general.run.general_run_setup import train_run_name, eval_run_name, Datamodule_selection, Channel_selection

from Contrastive_uncertainty.Contrastive.datamodules.datamodule_dict import dataset_dict
from Contrastive_uncertainty.Contrastive.callbacks.general_callbacks import  ModelSaving,SupConLoss,Uniformity,MMD_distance,Centroid_distance
from Contrastive_uncertainty.Contrastive.callbacks.ood_callbacks import  Mahalanobis_OOD, Euclidean_OOD  #,ImagePredictionLogger, OOD_ROC, OOD_confusion_matrix,
from Contrastive_uncertainty.Contrastive.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.Contrastive.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    val_train_loader, val_test_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    samples = next(iter(val_test_loader))
    sample_size = config['bsz']
    num_classes = config['num_classes']
    quick_callback = config['quick_callback']
    inference_clusters = [num_classes]
    OOD_val_train_loader, OOD_val_test_loader = OOD_Datamodule.val_dataloader()

    OOD_samples = next(iter(OOD_val_test_loader))

    callback_dict = {'Model_saving':ModelSaving(config['model_saving']), 
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_test_loader,evaltypes,config['quick_callback']),
                
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters, quick_callback=quick_callback),
                
                'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters,quick_callback=quick_callback),'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
                
                'Visualisation': Visualisation(Datamodule, OOD_Datamodule,num_classes,config['quick_callback']),'Uniformity': Uniformity(2, Datamodule, config['quick_callback']),
                
                'Centroid': Centroid_distance(Datamodule, config['quick_callback']), 'SupCon': SupConLoss(Datamodule, config['quick_callback'])}
    
    return callback_dict

#'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),