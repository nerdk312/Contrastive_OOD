
# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule,channels):
    model = model_module(emb_dim = config['emb_dim'], 
            optimizer = config['optimizer'],learning_rate = config['learning_rate'],
            momentum = config['momentum'], weight_decay = config['weight_decay'],
            datamodule = datamodule,num_classes = config['num_classes'],
            label_smoothing=config['label_smoothing'],num_channels = channels,
            instance_encoder = config['instance_encoder'],
            pretrained_network = config['pretrained_network'])

    return model