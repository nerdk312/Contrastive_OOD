from pytorch_lightning import callbacks

trainer_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 128, #32,#64,
OOD_dataset = ['MNIST', 'FashionMNIST', 'KMNIST'],
#OOD_dataset = ['STL10', 'CelebA','WIDERFace','SVHN', 'Caltech101','Caltech256','CIFAR10','CIFAR100', 'VOC', 'Places365','TinyImageNet','Cub200','Dogs', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST'],
#OOD_dataset = ['KMNIST'],
#OOD_dataset =['MNIST','FashionMNIST','KMNIST','EMNIST','Places365','VOC'],
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
callbacks = [],
#callbacks = ['Oracle Nearest 10 Neighbours Class 1D Typicality'],
#callbacks = ['Metrics'],
#callbacks = ['Different K Nearest Neighbours Class'],
#callbacks = ['Mahalanobis Distance','Different K Nearest Neighbours Class 1D Typicality','Nearest'],
#callbacks = ['Nearest 10 Neighbours Class Quadratic 1D Typicality'],

#callbacks = ['Mahalanobis Distance','Maximum Softmax Probability','ODIN'],
#callbacks = ['Nearest 10 Neighbours Class Quadratic 1D Typicality'],
#callbacks = ['Maximum Softmax Probability'],
#callbacks = ['Mahalanobis Distance','Maximum Softmax Probability'],
#callbacks = ['Maximum Softmax Probability','Nearest 10 Neighbours Class Quadratic 1D Typicality'],
#callbacks = ['Nearest 10 Neighbours Class Typicality'],
#callbacks = ['Gram'],
#callbacks = ['CAM'],
#callbacks = ['KDE'],
#callbacks = ['Nearest 10 Neighbours Marginal Quadratic 1D Typicality','Nearest 1 Neighbours Class Quadratic 1D Typicality'],
#callbacks = ['Nearest 10 Neighbours Class Typicality'],
#callbacks = ['Total Centroid KL'],
#callbacks = ['Feature Entropy'],
#callbacks = ['Mahalanobis Distance'],
#callbacks = ['KDE'],
#callbacks = ['Nearest 10 Neighbours Class Quadratic 1D Typicality'],
#callbacks = ['Nearest 10 Neighbours Class 1D Typicality','Nearest 10 Neighbours Class Quadratic 1D Typicality','Mahalanobis Distance'],
#callbacks = ['Nearest','Class Mahalanobis','Mahalanobis Distance'],

#callbacks = ['Nearest Neighbours Class 1D Typicality'],

#callbacks = ['Nearest Class Neighbours'],
#callbacks = ['Nearest Neighbours 1D Typicality'],
#callbacks = ['Class Mahalanobis','Mahalanobis OOD Fractions', 'Nearest Neighbours'],
#callbacks = ['Model_saving']
)

if 'Gram' in trainer_hparams['callbacks']:
    trainer_hparams['instance_encoder'] = 'gram_resnet50'
else:
    pass