from warnings import warn

from torchvision import transforms
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import \
    imagenet_normalization, cifar10_normalization,cifar100_normalization, stl10_normalization, \
    svhn_normalization, fashionmnist_normalization, mnist_normalization, kmnist_normalization,\
    emnist_normalization