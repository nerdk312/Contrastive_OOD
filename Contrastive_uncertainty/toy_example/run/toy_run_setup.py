from Contrastive_uncertainty.toy_example.callbacks.toy_visualisation_callbacks import  circular_visualisation, data_visualisation,\
                                                                               TwoMoonsUncertaintyVisualisation , TwoMoonsRepresentationVisualisation
from Contrastive_uncertainty.toy_example.callbacks.toy_general_callbacks import ModelSaving
from Contrastive_uncertainty.toy_example.callbacks.toy_ood_callbacks import OOD_ROC, Mahalanobis_OOD
from Contrastive_uncertainty.toy_example.datamodules.datamodule_dict import dataset_dict

from Contrastive_uncertainty.toy_example.models.toy_moco import MocoToy
from Contrastive_uncertainty.toy_example.models.toy_PCL import PCLToy
from Contrastive_uncertainty.toy_example.models.toy_softmax import SoftmaxToy
from Contrastive_uncertainty.toy_example.models.toy_supcon import SupConToy
from Contrastive_uncertainty.toy_example.models.toy_ova import OVAToy
from Contrastive_uncertainty.toy_example.models.toy_uniform import UniformityToy
from Contrastive_uncertainty.toy_example.models.toy_align import AlignmentToy
from Contrastive_uncertainty.toy_example.models.toy_align_uniform import AlignmentUniformityToy
from Contrastive_uncertainty.toy_example.models.toy_ova_uniform import OVAUniformityToy
from Contrastive_uncertainty.toy_example.models.toy_NNCL import NNCLToy
from Contrastive_uncertainty.toy_example.models.toy_ova_uniform_cluster import OVAUniformClusterToy
from Contrastive_uncertainty.toy_example.models.toy_multi_PCL import MultiPCLToy
from Contrastive_uncertainty.toy_example.models.toy_multi_PCL_branch import MultiPCLBranchToy

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
                     'Uncertainty_visualise': TwoMoonsUncertaintyVisualisation(Datamodule),
                     'Representation_visualise': TwoMoonsRepresentationVisualisation(Datamodule),
                     'ROC': OOD_ROC(Datamodule, OOD_Datamodule),
                     'Mahalanobis': Mahalanobis_OOD(Datamodule, OOD_Datamodule, config['quick_callback']),
                     'Saving': ModelSaving(1)}

    return callback_dict


def Model_selection(datamodule,config):
    model_dict = {'Softmax': SoftmaxToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes'],pretrained_network=config['pretrained_network']),

                'PCL':PCLToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network']),
                
                'MultiPCL':MultiPCLToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network']),#, num_cluster=config['num_cluster']),

                'MultiPCLBranch':MultiPCLBranchToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network']),
    }
    #import ipdb; ipdb.set_trace()
    
    '''
    model_dict = {'Moco':MocoToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network'], num_classes=config['num_classes']),
                
                'SupCon':SupConToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                softmax_temperature=config['softmax_temperature'],base_temperature=config['softmax_temperature'],
                num_classes= config['num_classes']),

                'PCL':PCLToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network']),#, num_cluster=config['num_cluster']),

                'NNCL':NNCLToy(datamodule=datamodule,
                optimizer=config['optimizer'], learning_rate=config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'], emb_dim=config['emb_dim'],
                num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
                softmax_temperature=config['softmax_temperature'],
                pretrained_network=config['pretrained_network']),#, num_cluster=config['num_cluster']),),

                'Softmax': SoftmaxToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),
                
                'OVA': OVAToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),
                
                'Uniformity':UniformityToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),

                'Alignment':AlignmentToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),

                'AlignUniform':AlignmentUniformityToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),

                'OVAUniform': OVAUniformityToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes']),

                'OVAUniformCluster': OVAUniformClusterToy(datamodule=datamodule,
                optimizer= config['optimizer'],learning_rate= config['learning_rate'],
                momentum=config['momentum'], weight_decay=config['weight_decay'],
                hidden_dim=config['hidden_dim'],emb_dim=config['emb_dim'],
                num_classes = config['num_classes'],encoder_momentum=config['encoder_momentum'],
                softmax_temperature = config['softmax_temperature'],
                num_cluster = config['num_cluster']),

    }
    '''
    return model_dict[config['model']]