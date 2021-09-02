from Contrastive_uncertainty.general.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.general.datamodules.cifar100_datamodule import CIFAR100DataModule
from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.general.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general.datamodules.kmnist_datamodule import KMNISTDataModule
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule
from Contrastive_uncertainty.general.datamodules.svhn_datamodule import SVHNDataModule
from Contrastive_uncertainty.general.datamodules.stl10_datamodule import STL10DataModule
from Contrastive_uncertainty.general.datamodules.emnist_datamodule import EMNISTDataModule


from Contrastive_uncertainty.general.datamodules.datamodule_transforms import  Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms,Moco2MultiCIFAR10Transforms,\
Moco2TrainCIFAR100Transforms, Moco2EvalCIFAR100Transforms,Moco2MultiCIFAR100Transforms,\
Moco2TrainFashionMNISTTransforms,Moco2EvalFashionMNISTTransforms, Moco2MultiFashionMNISTTransforms, \
Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms,Moco2MultiMNISTTransforms,\
Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms,Moco2MultiSVHNTransforms,\
Moco2TrainKMNISTTransforms,Moco2EvalKMNISTTransforms,Moco2MultiKMNISTTransforms, \
Moco2TrainSTL10Transforms, Moco2EvalSTL10Transforms, Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms, Moco2MultiEMNISTTransforms


# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2TrainMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms(),'multi_transform':Moco2MultiMNISTTransforms},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms(), 'multi_transform':Moco2MultiKMNISTTransforms},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2TrainFashionMNISTTransforms(),
                'val_transform':Moco2EvalFashionMNISTTransforms(),'test_transform':Moco2EvalFashionMNISTTransforms(), 'multi_transform':Moco2MultiFashionMNISTTransforms},

                'EMNIST':{'module':EMNISTDataModule,'train_transform':Moco2TrainEMNISTTransforms(),
                'val_transform':Moco2EvalEMNISTTransforms(),'test_transform':Moco2EvalEMNISTTransforms(), 'multi_transform':Moco2MultiEMNISTTransforms},

                'CIFAR10':{'module':CIFAR10DataModule,'train_transform':Moco2TrainCIFAR10Transforms(),
                'val_transform':Moco2EvalCIFAR10Transforms(),'test_transform':Moco2EvalCIFAR10Transforms(),'multi_transform':Moco2MultiCIFAR10Transforms},
                
                'CIFAR100':{'module':CIFAR100DataModule,'train_transform':Moco2TrainCIFAR100Transforms(),
                'val_transform':Moco2EvalCIFAR100Transforms(),'test_transform':Moco2EvalCIFAR100Transforms(), 'multi_transform':Moco2MultiCIFAR100Transforms},

                'STL10':{'module': STL10DataModule,'train_transform':Moco2TrainSTL10Transforms(),
                'val_transform':Moco2EvalSTL10Transforms(),'test_transform':Moco2EvalSTL10Transforms()},

                'SVHN':{'module':SVHNDataModule,'train_transform':Moco2TrainSVHNTransforms(),
                'val_transform':Moco2EvalSVHNTransforms(),'test_transform':Moco2EvalSVHNTransforms(), 'multi_transform':Moco2MultiSVHNTransforms}
                
                }



'''
dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2EvalMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms()},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms()},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2EvalFashionMNISTTransforms(),
                'val_transform':Moco2EvalFashionMNISTTransforms(),'test_transform':Moco2EvalFashionMNISTTransforms()},

                'EMNIST':{'module':EMNISTDataModule,'train_transform':Moco2TrainEMNISTTransforms(),
                'val_transform':Moco2EvalEMNISTTransforms(),'test_transform':Moco2EvalEMNISTTransforms()},

                'CIFAR10':{'module':CIFAR10DataModule,'train_transform':Moco2TrainCIFAR10Transforms(),
                'val_transform':Moco2EvalCIFAR10Transforms(),'test_transform':Moco2EvalCIFAR10Transforms()},
                
                'CIFAR100':{'module':CIFAR100DataModule,'train_transform':Moco2TrainCIFAR100Transforms(),
                'val_transform':Moco2EvalCIFAR100Transforms(),'test_transform':Moco2EvalCIFAR100Transforms()},

                'STL10':{'module': STL10DataModule,'train_transform':Moco2TrainSTL10Transforms(),
                'val_transform':Moco2EvalSTL10Transforms(),'test_transform':Moco2EvalSTL10Transforms()},

                 'SVHN':{'module':SVHNDataModule,'train_transform':Moco2TrainSVHNTransforms(),
                'val_transform':Moco2EvalSVHNTransforms(),'test_transform':Moco2EvalSVHNTransforms()}
                
                }

'''


OOD_dict = {'MNIST':['FashionMNIST','KMNIST','EMNIST'],
            'FashionMNIST':['MNIST','KMNIST','EMNIST'],
            'KMNIST':['MNIST','FashionMNIST','EMNIST'],
            'EMNIST':['MNIST','FashionMNIST','KMNIST'],

            'CIFAR10':['CIFAR100','SVHN'],
            'CIFAR100':['CIFAR10','SVHN'],
            'SVHN':['CIFAR10','CIFAR100']}