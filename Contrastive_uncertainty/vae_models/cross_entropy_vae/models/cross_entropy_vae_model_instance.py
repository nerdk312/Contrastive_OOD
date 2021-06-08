# Function which instantiates cross entropy model
from Contrastive_uncertainty.general.utils.hybrid_utils import label_smoothing


def ModelInstance(model_module,config,datamodule):
    model = model_module(emb_dim = config['emb_dim'], 
            optimizer = config['optimizer'],learning_rate = config['learning_rate'],
            momentum = config['momentum'], weight_decay = config['weight_decay'],
            datamodule = datamodule, pretrained_network = config['pretrained_network'],
            instance_encoder = config['instance_encoder'],first_conv = config['first_conv'],
            maxpool1 = config['maxpool1'], enc_out_dim = config['enc_out_dim'],
            kl_coeff = config['kl_coeff'],label_smoothing = config['label_smoothing'])

    return model