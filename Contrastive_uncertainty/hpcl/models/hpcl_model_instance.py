# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule,channels):


    model = model_module(datamodule=datamodule, optimizer=config['optimizer'],
    learning_rate=config['learning_rate'], momentum=config['momentum'],
    weight_decay=config['weight_decay'], emb_dim=config['emb_dim'],
    num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
    softmax_temperature=config['softmax_temperature'], 
    use_mlp=config['use_mlp'],instance_encoder=config['instance_encoder'], pretrained_network=config['pretrained_network'])

    return model
