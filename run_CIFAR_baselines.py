from Contrastive_uncertainty.cross_entropy.train.train_cross_entropy import train
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params  import cross_entropy_hparams

print('First test CE')
cross_entropy_hparams['epochs'] = 300
cross_entropy_hparams['dataset'] = 'CIFAR10'
cross_entropy_hparams['OOD_dataset'] = 'SVHN'
# calls the function
train(cross_entropy_hparams)


print('Second test SUPCON')
from Contrastive_uncertainty.sup_con.train.train_sup_con import train
from Contrastive_uncertainty.sup_con.config.sup_con_params import sup_con_hparams

sup_con_hparams['epochs'] = 300
sup_con_hparams['dataset'] = 'CIFAR10'
sup_con_hparams['OOD_dataset'] = 'SVHN'
# calls the function
train(sup_con_hparams)

print('Third test Moco')
from Contrastive_uncertainty.Contrastive.train.train_contrastive import train
from Contrastive_uncertainty.Contrastive.config.contrastive_params import contrastive_hparams

contrastive_hparams['epochs'] = 300
contrastive_hparams['dataset'] = 'CIFAR10'
contrastive_hparams['OOD_dataset'] = 'SVHN'
# calls the function
train(contrastive_hparams)



print('Fourth test PCL')
from Contrastive_uncertainty.PCL.train.train_pcl import training
from Contrastive_uncertainty.PCL.config.pcl_params import pcl_hparams

pcl_hparams['epochs'] = 300
pcl_hparams['dataset'] = 'CIFAR10'
pcl_hparams['OOD_dataset'] = 'SVHN'

training(pcl_hparams)


print('Fifth test Unsupcon memory')
from Contrastive_uncertainty.unsup_con_memory.train.train_unsup_con_memory import train
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams
unsup_con_memory_hparams['epochs'] = 300
unsup_con_memory_hparams['dataset'] = 'CIFAR10'
unsup_con_memory_hparams['OOD_dataset'] = 'SVHN'
# calls the function
train(unsup_con_memory_hparams)