# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule,channels,num_clusters):
    model = model_module(datamodule=datamodule, optimizer=config['optimizer'],
    learning_rate=config['learning_rate'], momentum=config['momentum'],
    weight_decay=config['weight_decay'], emb_dim=config['emb_dim'],
    num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
    softmax_temperature=config['softmax_temperature'], 
    num_cluster=num_clusters,num_cluster_negatives=config['num_cluster_negatives'],
    use_mlp=config['use_mlp'],num_channels=channels, 
    instance_encoder=config['instance_encoder'], pretrained_network=config['pretrained_network'])

    return model
