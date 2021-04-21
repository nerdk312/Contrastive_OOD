from warnings import warn

from torchvision import transforms

from Contrastive_uncertainty.PCL.datamodules.dataset_normalizations import \
    imagenet_normalization, cifar10_normalization, stl10_normalization, \
    fashionmnist_normalization, mnist_normalization, svhn_normalization
