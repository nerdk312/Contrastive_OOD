from Contrastive_uncertainty.general.callbacks.general_callbacks import  ModelSaving,SupConLoss,Uniformity,MMD_distance,Centroid_distance
from Contrastive_uncertainty.general.callbacks.ood_callbacks import  Mahalanobis_OOD, Euclidean_OOD#, IsoForest
from Contrastive_uncertainty.general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes

from Contrastive_uncertainty.general.run.general_run_setup import train_run_name, eval_run_name,\
    Datamodule_selection #Channel_selection


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    sample_size = config['bsz']
    num_classes = Datamodule.num_classes
    quick_callback = config['quick_callback']
    inference_clusters = config['num_inference_cluster']
    val_loader= Datamodule.val_dataloader() # Used for metric logger callback also
    samples = next(iter(val_loader))
    OOD_val_loader = OOD_Datamodule.val_dataloader()
    OOD_samples = next(iter(OOD_val_loader))

    callback_dict = {'Model_saving':ModelSaving(config['model_saving']), 
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_loader,evaltypes,quick_callback=quick_callback),
                
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters, quick_callback=quick_callback),
                
                'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule,num_inference_clusters=inference_clusters,quick_callback=quick_callback),'MMD': MMD_distance(Datamodule, quick_callback=quick_callback),
                
                'Visualisation': Visualisation(Datamodule, OOD_Datamodule,quick_callback=quick_callback),'Uniformity': Uniformity(2, Datamodule, quick_callback=quick_callback),
                
                'Centroid': Centroid_distance(Datamodule, quick_callback=quick_callback), 'SupCon': SupConLoss(Datamodule, quick_callback=quick_callback)}
    
    return callback_dict

#'IsoForest': IsoForest(Datamodule,OOD_Datamodule, quick_callback=quick_callback),