# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule,channels):
    # Hack to be able to use the num clusters with wandb sweep since wandb sweep cannot use a list of lists I believe
    if isinstance(config['num_multi_cluster'], list) or isinstance(config['num_multi_cluster'], tuple):
        num_clusters = config['num_multi_cluster']
    else:  
        num_clusters = [config['num_multi_cluster']]

    model = model_module(datamodule=datamodule, optimizer=config['optimizer'],
    learning_rate=config['learning_rate'], momentum=config['momentum'],
    weight_decay=config['weight_decay'], emb_dim=config['emb_dim'],
    num_negatives=config['num_negatives'], encoder_momentum=config['encoder_momentum'],
    softmax_temperature=config['softmax_temperature'], 
    num_cluster=num_clusters,
    use_mlp=config['use_mlp'],num_channels=channels, 
    instance_encoder=config['instance_encoder'], pretrained_network=config['pretrained_network'])

    return model
