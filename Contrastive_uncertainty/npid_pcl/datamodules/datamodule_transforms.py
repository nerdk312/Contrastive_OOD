import random
from warnings import warn



from PIL import ImageFilter
from torchvision import transforms
from Contrastive_uncertainty.npid_pcl.datamodules.dataset_normalizations import \
    cifar10_normalization, fashionmnist_normalization,mnist_normalization,svhn_normalization

from Contrastive_uncertainty.general_clustering.datamodules.datamodule_transforms import \
    Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, \
    Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms, \
    Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, \
    Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, \
    GaussianBlur, dataset_with_indices