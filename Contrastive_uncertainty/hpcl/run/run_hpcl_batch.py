from Contrastive_uncertainty.PCL.train.train_pcl import training
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams

# calls the function
training(pcl_hparams)

print('Second Run')
pcl_hparams['dataset'] = 'MNIST' 
pcl_hparams['OOD_dataset'] = 'FashionMNIST'
training(pcl_hparams)

print('Third Run')
pcl_hparams['dataset'] = 'KMNIST' 
pcl_hparams['OOD_dataset'] = 'MNIST'
training(pcl_hparams)
