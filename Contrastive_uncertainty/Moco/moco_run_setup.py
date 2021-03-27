from Contrastive_uncertainty.datamodules.datamodule_dict import dataset_dict
from Contrastive_uncertainty.Moco.moco_callbacks import ReliabiltyLogger, ImagePredictionLogger, ModelSaving, OOD_confusion_matrix, OOD_ROC, \
                                                        Mahalanobis_OOD, Mahalanobis_OOD_compressed, Euclidean_OOD
from Contrastive_uncertainty.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes

def run_name(config):
    run_name = 'Epochs:'+ str(config['epochs']) +  '_lr:' + f"{config['learning_rate']:.3e}" + '_bsz:' + str(config['bsz']) +  '_classifier:' +str(config['classifier']) + '_seed:' +str(config['seed'])  
    return run_name

def Evaluation_run_name(config):
    run_name = 'Evaluation_' +'Epochs:'+ str(config['epochs']) +  '_lr:' + f"{config['learning_rate']:.3e}" + '_bsz:' + str(config['bsz']) +  '_classifier:' +str(config['classifier']) + '_seed:' +str(config['seed'])  
    return run_name

def Datamodule_selection(dataset,config):
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

    OOD_val_train_loader, OOD_val_test_loader = OOD_Datamodule.val_dataloader()

    OOD_samples = next(iter(OOD_val_test_loader))

    callback_dict = {'Model_saving':ModelSaving(config['model_saving']), 'Confusion_matrix':OOD_confusion_matrix(Datamodule,OOD_Datamodule),'ROC':OOD_ROC(Datamodule,OOD_Datamodule),
                'Reliability': ReliabiltyLogger(samples,sample_size), 'Metrics': MetricLogger(evaluation_metrics,val_test_loader,evaltypes,config['quick_callback']),'Image_prediction':ImagePredictionLogger(samples,OOD_samples,sample_size),
                'Mahalanobis': Mahalanobis_OOD(Datamodule,OOD_Datamodule, config['quick_callback']), 'Mahalanobis_compressed': Mahalanobis_OOD_compressed(Datamodule,OOD_Datamodule,config['quick_callback']),
                'Euclidean':Euclidean_OOD(Datamodule,OOD_Datamodule, config['quick_callback'])}
    
    return callback_dict
