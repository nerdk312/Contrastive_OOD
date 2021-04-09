from Contrastive_uncertainty.toy_example.callbacks.toy_visualisation_callbacks import circular_visualisation, data_visualisation,\
                                                                               UncertaintyVisualisation , TwoMoonsVisualisation    
from Contrastive_uncertainty.toy_example.callbacks.toy_ood_callbacks import OOD_ROC, Mahalanobis_OOD

def callback_dictionary(Datamodule,OOD_Datamodule,config):
    callback_dict = {'Circular_visualise': circular_visualisation(Datamodule), 
                     'Data_visualise': data_visualisation(Datamodule, OOD_Datamodule),
                     'Uncertainty_visualise': UncertaintyVisualisation(Datamodule),
                     'ROC': OOD_ROC(Datamodule, OOD_Datamodule),
                     'Mahalanobis': Mahalanobis_OOD(Datamodule, OOD_Datamodule, config['quick_callback'])}


    return callback_dict