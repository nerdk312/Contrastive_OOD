from Contrastive_uncertainty.toy_replica.toy_general.callbacks.general_callbacks import ModelSaving, MMD_distance
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.ood_callbacks import Mahalanobis_OOD
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.toy_replica.toy_general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes


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
    datamodule_info =  data_dict[dataset] # Specific module
    #import ipdb; ipdb.set_trace()
    Datamodule = datamodule_info['module'](data_dir= './',batch_size = config['bsz'],seed = config['seed'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    val_loader= Datamodule.val_dataloader() # Used for metric logger callback also
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

def specific_callbacks(callback_dict, names):
    desired_callbacks = []    
    for index, name in enumerate(names):
    #for index, name in enumerate(config['callbacks']):
        #import ipdb; ipdb.set_trace()
        desired_callbacks.append(callback_dict[name])

    return desired_callbacks
