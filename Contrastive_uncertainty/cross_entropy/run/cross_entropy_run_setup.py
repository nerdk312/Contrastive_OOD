from Contrastive_uncertainty.cross_entropy.datamodules.datamodule_dict import dataset_dict

from Contrastive_uncertainty.cross_entropy.callbacks.general_callbacks import  ModelSaving,SupConLoss,Uniformity,MMD_distance,Centroid_distance
from Contrastive_uncertainty.cross_entropy.callbacks.ood_callbacks import  Mahalanobis_OOD, Euclidean_OOD  #,ImagePredictionLogger, OOD_ROC, OOD_confusion_matrix,
from Contrastive_uncertainty.cross_entropy.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.cross_entropy.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes


def run_name(config):
    run_name = 'Epochs:'+ str(config['epochs']) +  '_lr:' + f"{config['learning_rate']:.3e}" + '_bsz:' + str(config['bsz']) + '_seed:' +str(config['seed'])  
    return run_name

def Evaluation_run_name(config):
    run_name = 'Evaluation_' +'Epochs:'+ str(config['epochs']) +  '_lr:' + f"{config['learning_rate']:.3e}" + '_bsz:' + str(config['bsz']) + '_seed:' +str(config['seed'])  
    return run_name

def Datamodule_selection(dataset, config):
    # Information regarding the configuration of the data module for the specific task
    datamodule_info =  dataset_dict[dataset] # Specific module
    Datamodule = datamodule_info['module'](data_dir= './',batch_size = config['bsz'],seed = config['seed'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule

def Channel_selection(dataset):
    datamodule_info =  dataset_dict[dataset]
    channels = datamodule_info['channels']
    return channels


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    val_train_loader, val_test_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    samples = next(iter(val_test_loader))
    sample_size = config['bsz']
    num_classes = config['num_classes']
    OOD_val_train_loader, OOD_val_test_loader = OOD_Datamodule.val_dataloader()

    OOD_samples = next(iter(OOD_val_test_loader))

    callback_dict = {'Model_saving':ModelSaving(config['model_saving']), 
                'Metrics': MetricLogger(evaluation_metrics,num_classes,val_test_loader,evaltypes,config['quick_callback']),
                
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule, config['quick_callback']),
                
                'Euclidean': Euclidean_OOD(Datamodule,OOD_Datamodule, config['quick_callback']),'MMD': MMD_distance(Datamodule, config['quick_callback']),
                
                'Visualisation': Visualisation(Datamodule, OOD_Datamodule,num_classes,config['quick_callback']),'Uniformity': Uniformity(2, Datamodule, config['quick_callback']),
                
                'Centroid': Centroid_distance(Datamodule, config['quick_callback']), 'SupCon': SupConLoss(Datamodule, config['quick_callback'])}
    
    return callback_dict

#'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),