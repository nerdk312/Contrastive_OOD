from Contrastive_uncertainty.general_hierarchy.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.cifar100_datamodule import CIFAR100DataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.kmnist_datamodule import KMNISTDataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.emnist_datamodule import EMNISTDataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.svhn_datamodule import SVHNDataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.stl10_datamodule import STL10DataModule
from Contrastive_uncertainty.general_hierarchy.datamodules.emnist_datamodule import EMNISTDataModule


from Contrastive_uncertainty.general_hierarchy.datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms,\
Moco2TrainCIFAR100Transforms, Moco2EvalCIFAR100Transforms, Moco2TrainFashionMNISTTransforms,Moco2EvalFashionMNISTTransforms, \
Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms, Moco2TrainKMNISTTransforms,Moco2EvalKMNISTTransforms, \
Moco2TrainSTL10Transforms, Moco2EvalSTL10Transforms, Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms


# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary
dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2TrainMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms()},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms()},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2TrainFashionMNISTTransforms(),
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
