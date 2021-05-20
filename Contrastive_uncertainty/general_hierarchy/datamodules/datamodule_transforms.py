import numpy as np
import random
from warnings import warn
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import cifar10_normalization,\
    cifar100_normalization, fashionmnist_normalization, mnist_normalization, kmnist_normalization,\
    svhn_normalization, stl10_normalization, emnist_normalization
    


from PIL import ImageFilter
from torchvision import transforms

from Contrastive_uncertainty.general.datamodules.datamodule_transforms import \
    Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, \
    Moco2TrainCIFAR100Transforms, Moco2EvalCIFAR100Transforms, \
    Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms,\
    Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, \
    Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, \
    Moco2TrainKMNISTTransforms, Moco2EvalKMNISTTransforms, \
    Moco2TrainSTL10Transforms, Moco2EvalSTL10Transforms, \
    Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms,\
    GaussianBlur, dataset_with_indices, dataset_with_indices_hierarchy, \
    dataset_with_indices_emnist, \
    MNIST_coarse_labels, CIFAR10_coarse_labels, CIFAR100_coarse_labels
