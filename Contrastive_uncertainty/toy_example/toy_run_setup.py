from Contrastive_uncertainty.toy_example.toy_callbacks import circular_visualisation, data_visualisation,  OOD_ROC

def callback_dictionary(Datamodule,OOD_Datamodule,config):
    
    callback_dict = {'Circular_visualise': circular_visualisation(Datamodule), 
                     'Data_visualise': data_visualisation(Datamodule, OOD_Datamodule),
                     'ROC': OOD_ROC(Datamodule, OOD_Datamodule)}

    
    return callback_dict