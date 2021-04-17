from Contrastive_uncertainty.toy_NCA.callbacks.toy_visualisation_callbacks import circular_visualisation, data_visualisation,\
                                                                               UncertaintyVisualisation , TwoMoonsVisualisation    
from Contrastive_uncertainty.toy_NCA.callbacks.toy_ood_callbacks import OOD_ROC, Mahalanobis_OOD
from Contrastive_uncertainty.toy_NCA.datamodules.datamodule_dict import dataset_dict

from Contrastive_uncertainty.toy_NCA.models.toy_nca import NCAToy


def Datamodule_selection(dataset, config):
    # Information regarding the configuration of the data module for the specific task
    datamodule_info =  dataset_dict[dataset] # Specific module
    Datamodule = datamodule_info['module'](batch_size = config['bsz'])
    Datamodule.train_transforms = datamodule_info['train_transform']
    Datamodule.val_transforms = datamodule_info['val_transform']
    Datamodule.test_transforms = datamodule_info['test_transform']
    Datamodule.prepare_data()
    Datamodule.setup()
    return Datamodule


def callback_dictionary(Datamodule,OOD_Datamodule,config):
    callback_dict = {'Circular_visualise': circular_visualisation(Datamodule), 
                     'Data_visualise': data_visualisation(Datamodule, OOD_Datamodule),
                     'Uncertainty_visualise': TwoMoonsVisualisation(Datamodule),
                     'ROC': OOD_ROC(Datamodule, OOD_Datamodule),
                     'Mahalanobis': Mahalanobis_OOD(Datamodule, OOD_Datamodule, config['quick_callback'])}


    return callback_dict


def Model_selection(datamodule,config):
    data_labels = datamodule.train_dataloader().dataset.tensors[1]
    model_dict = {'NCA': NCAToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes'],memory_momentum=config['memory_momentum'],
                softmax_temperature = config['softmax_temperature'],
                labels=data_labels),

    }
    
    return model_dict[config['model']]