# Function which instantiates cross entropy model
def ModelInstance(model_module,config,datamodule,channels):
    # Hack to be able to use the num clusters with wandb sweep since wandb sweep cannot use a list of lists I believe
    if isinstance(config['num_cluster'], list) or isinstance(config['num_cluster'], tuple):
        num_clusters = config['num_cluster']
    else:  
        num_clusters = [config['num_cluster']]

    model = model_module(emb_dim = config['emb_dim'],num_negatives = config['num_negatives'],
        memory_momentum = config['memory_momentum'],num_cluster=num_clusters, 
        softmax_temperature = config['softmax_temperature'],
        optimizer = config['optimizer'],learning_rate = config['learning_rate'],
        momentum = config['momentum'], weight_decay = config['weight_decay'],
        use_mlp = config['use_mlp'],
        datamodule = datamodule,num_channels = channels,
        instance_encoder = config['instance_encoder'],
        pretrained_network = config['pretrained_network'])

    return model
