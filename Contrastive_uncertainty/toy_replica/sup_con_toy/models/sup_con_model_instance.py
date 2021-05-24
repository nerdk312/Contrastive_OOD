# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule):
    model = model_module(emb_dim = config['emb_dim'],contrast_mode=config['contrast_mode'],
            softmax_temperature = config['softmax_temperature'],
            optimizer = config['optimizer'],learning_rate = config['learning_rate'],
            momentum = config['momentum'], weight_decay = config['weight_decay'],
            datamodule = datamodule)
    
    return model