from Contrastive_uncertainty.imp.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
from Contrastive_uncertainty.imp.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.imp.datamodules.cifar10_datamodule import CIFAR10DataModule
from Contrastive_uncertainty.imp.datamodules.svhn_datamodule import SVHNDataModule
from Contrastive_uncertainty.imp.datamodules.kmnist_datamodule import KMNISTDataModule

from Contrastive_uncertainty.imp.datamodules.datamodule_transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms,\
Moco2TrainFashionMNISTTransforms,Moco2EvalFashionMNISTTransforms, Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, \
Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms, Moco2TrainKMNISTTransforms,Moco2EvalKMNISTTransforms


# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary
dataset_dict = {'MNIST':{'module':MNISTDataModule,'train_transform':Moco2TrainMNISTTransforms(),'val_transform':Moco2EvalMNISTTransforms(),'test_transform':Moco2EvalMNISTTransforms(),'channels':1},
                
                'KMNIST':{'module':KMNISTDataModule,'train_transform':Moco2TrainKMNISTTransforms(),
                'val_transform':Moco2EvalKMNISTTransforms(),'test_transform':Moco2EvalKMNISTTransforms(),'channels':1},

                'FashionMNIST':{'module':FashionMNISTDataModule,'train_transform':Moco2TrainFashionMNISTTransforms(),
                'val_transform':Moco2EvalFashionMNISTTransforms(),'test_transform':Moco2EvalFashionMNISTTransforms(),'channels':1},

                'CIFAR10':{'module':CIFAR10DataModule,'train_transform':Moco2TrainCIFAR10Transforms(),
                'val_transform':Moco2EvalCIFAR10Transforms(),'test_transform':Moco2EvalCIFAR10Transforms(),'channels':3},                
                
                'SVHN':{'module':SVHNDataModule,'train_transform':Moco2TrainSVHNTransforms(),
                'val_transform':Moco2EvalSVHNTransforms(),'test_transform':Moco2EvalSVHNTransforms(),'channels':3}

                }

