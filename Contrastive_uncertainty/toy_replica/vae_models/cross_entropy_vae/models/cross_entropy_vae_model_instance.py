# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule):
    model = model_module(emb_dim = config['emb_dim'], 
            optimizer = config['optimizer'],learning_rate = config['learning_rate'],
            momentum = config['momentum'], weight_decay = config['weight_decay'],
            datamodule = datamodule,label_smoothing=config['label_smoothing'],
            kl_coeff = config['kl_coeff'], pretrained_network = config['pretrained_network'])

    return model
