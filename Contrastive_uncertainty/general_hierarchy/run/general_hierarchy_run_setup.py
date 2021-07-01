from Contrastive_uncertainty.general.callbacks.practice.practice_hierarchical_callback import Practice_Hierarchical
from Contrastive_uncertainty.general.callbacks.general_callbacks import  ModelSaving, MMD_distance
from Contrastive_uncertainty.general.callbacks.ood_callbacks import Mahalanobis_OOD, Mahalanobis_OOD_Datasets, Mahalanobis_OvO, Mahalanobis_OvR, Mahalanobis_Subsample
from Contrastive_uncertainty.general.callbacks.experimental_ood_callbacks import  Aggregated_Mahalanobis_OOD, Differing_Mahalanobis_OOD 
from Contrastive_uncertainty.general.callbacks.visualisation_callback import Visualisation
from Contrastive_uncertainty.general.callbacks.typicality_ood_callback import Typicality_OVR, Typicality_OVO
from Contrastive_uncertainty.general.callbacks.metrics.metric_callback import MetricLogger, evaluation_metrics, evaltypes
from Contrastive_uncertainty.general.callbacks.variational_callback import Variational
from Contrastive_uncertainty.general.callbacks.hierarchical_ood import Hierarchical_Mahalanobis, Hierarchical_scores_comparison, Hierarchical_Subsample, Hierarchical_Relative_Mahalanobis
from Contrastive_uncertainty.general.callbacks.scores_callback import scores_comparison
#from Contrastive_uncertainty.general.callbacks.practice_callback import Comparison_practice
from Contrastive_uncertainty.general.callbacks.oracle_hierarchical_ood import Oracle_Hierarchical
from Contrastive_uncertainty.general.callbacks.practice.practice_hierarchical_callback import Practice_Hierarchical, Practice_Hierarchical_scores
from Contrastive_uncertainty.general.run.general_run_setup import Datamodule_selection, specific_callbacks
from Contrastive_uncertainty.general.callbacks.relative_mahalanobis_callback import One_Dim_Mahalanobis, Relative_Mahalanobis


# Run name which includes the branch weights
def train_run_name(model_name, config, group=None):
    run_name = "Train_" + model_name + "_DS:"+str(config["dataset"]) +"_Epochs:" + str(config["epochs"]) + "_seed:" +str(config["seed"]) + f'_instance:{config["branch_weights"][0]:.2f}_fine:{config["branch_weights"][1]:.2f}_coarse:{config["branch_weights"][2]:.2f}' 
    if group is not None:
        run_name = group + '_' + run_name
    return run_name

# Run name which includes the branch weights
def eval_run_name(model_name, config, group=None):
    run_name = "Eval_" + model_name + "_DS:"+str(config["dataset"]) +"_Epochs:" + str(config["epochs"]) + "_seed:" +str(config["seed"]) + f'_instance:{config["branch_weights"][0]:.2f}_fine:{config["branch_weights"][1]:.2f}_coarse:{config["branch_weights"][2]:.2f}' 
    if group is not None:
        run_name = group + '_' + run_name
    return run_name
    
    
# Generates the callbacks
def callback_dictionary(Datamodule,config,data_dict):
    #val_loader = Datamodule.val_dataloader() # Used for metric logger callback also
    #num_classes = Datamodule.num_classes
    
    quick_callback = config['quick_callback']
    typicality_batch = config['typicality_batch']
    typicality_bootstrap = config['typicality_bootstrap']
    vector_level = config['vector_level']
    label_level = config['label_level']

    # Manually added callbacks
    callback_dict = {'Model_saving':ModelSaving(config['model_saving'],'Models'),
                    'MMD_instance': MMD_distance(Datamodule,vector_level='instance', quick_callback=quick_callback),
                    'Metrics_instance_fine':MetricLogger(evaluation_metrics,Datamodule,evaltypes, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                    'Visualisation_instance_fine': Visualisation(Datamodule, vector_level='instance',label_level='fine',quick_callback=quick_callback),
                    'Variational':Variational(Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback)}
                    
    # Automatically adding callbacks for the Mahalanobis distance for each different vector level as well as each different OOD dataset
    # Collated list of OOD datamodules
    Collated_OOD_datamodules = []
    for ood_dataset in config['OOD_dataset']:
        OOD_Datamodule = Datamodule_selection(data_dict, ood_dataset, config)
        OOD_callback = {f'Mahalanobis {vector_level} {label_level} {ood_dataset}':Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level=vector_level, label_level=label_level),
                f'Aggregated {ood_dataset}': Aggregated_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Differing {ood_dataset}': Differing_Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'Typicality_OVR_{ood_dataset}': Typicality_OVR(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                f'Typicality_OVO_{ood_dataset}': Typicality_OVO(Datamodule,OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback,bootstrap_num=typicality_bootstrap,typicality_bsz=typicality_batch),
                f'OVR classification {ood_dataset}':Mahalanobis_OvR(Datamodule, OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                f'OVO classification {ood_dataset}':Mahalanobis_OvO(Datamodule, OOD_Datamodule, vector_level='instance', label_level='fine', quick_callback=quick_callback),
                f'One Dimensional Mahalanobis {ood_dataset}': One_Dim_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Relative Mahalanobis {ood_dataset}': Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Hierarchical {ood_dataset}':Hierarchical_Mahalanobis(Datamodule, OOD_Datamodule,quick_callback=quick_callback),
                f'Hierarchical Relative Mahalanobis {ood_dataset}': Hierarchical_Relative_Mahalanobis(Datamodule,OOD_Datamodule, quick_callback=quick_callback),
                f'Hierarchical Scores {ood_dataset}':Hierarchical_scores_comparison(Datamodule, OOD_Datamodule,quick_callback=quick_callback),
                f'Oracle Hierarchical {ood_dataset}':Oracle_Hierarchical(Datamodule, OOD_Datamodule,quick_callback=quick_callback),
                f'Practice Hierarchical scores {ood_dataset}': Practice_Hierarchical_scores(Datamodule,OOD_Datamodule,quick_callback=quick_callback),
                f'General Scores {ood_dataset}':scores_comparison(Datamodule,OOD_Datamodule,vector_level='coarse',label_level='coarse', quick_callback=quick_callback),
                f'Subsample': Hierarchical_Subsample(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}

                
                #f'Comparison {ood_dataset}': Comparison_practice(Datamodule, OOD_Datamodule, quick_callback=quick_callback)}
                
                #f'Subsample': Mahalanobis_Subsample(Datamodule,OOD_Datamodule,quick_callback=quick_callback)}
               
        callback_dict.update(OOD_callback)
        Collated_OOD_datamodules.append(OOD_Datamodule)
        '''
        vector_level = config['vector_level']
        label_level = config['label_level']
        for i in range(len(config['vector_level'])):
            mahalanobis_callback = {f'Mahalanobis {vector_level} {label_level} {ood_dataset}':Mahalanobis_OOD(Datamodule,OOD_Datamodule,quick_callback=quick_callback,vector_level=vector_level[i], label_level=label_level[i])}
            callback_dict.update(mahalanobis_callback)
        '''
    callback_dict.update({'OOD_Dataset_distances': Mahalanobis_OOD_Datasets(Datamodule, Collated_OOD_datamodules, quick_callback=quick_callback)})
    return callback_dict
    
