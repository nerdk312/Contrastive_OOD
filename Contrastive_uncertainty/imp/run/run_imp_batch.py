from Contrastive_uncertainty.imp.train.train_imp import training
from Contrastive_uncertainty.imp.config.imp_params import imp_hparams

# calls the function
training(imp_hparams)

print('Second Run')
imp_hparams['dataset'] = 'MNIST' 
imp_hparams['OOD_dataset'] = 'FashionMNIST'
training(imp_hparams)

print('Third Run')
imp_hparams['dataset'] = 'KMNIST' 
imp_hparams['OOD_dataset'] = 'MNIST'
training(imp_hparams)
