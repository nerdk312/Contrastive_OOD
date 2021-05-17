import random
import math
from warnings import warn


from PIL import ImageFilter
from torchvision import transforms

from Contrastive_uncertainty.general_clustering.datamodules.dataset_normalizations import  cifar10_normalization,\
    fashionmnist_normalization,kmnist_normalization,mnist_normalization,svhn_normalization,\
    emnist_normalization

from Contrastive_uncertainty.general.datamodules.datamodule_transforms import \
    Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, \
    Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms,\
    Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, \
    Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, \
    Moco2TrainKMNISTTransforms, Moco2EvalKMNISTTransforms, \
    Moco2TrainSTL10Transforms, Moco2EvalSTL10Transforms, \
    Moco2TrainEMNISTTransforms, Moco2EvalEMNISTTransforms,\
    GaussianBlur, dataset_with_indices, dataset_with_indices_hierarchy


def split_size(batch_size, samples): # obtains a dataset size for the k-means based on the batch size
        batch_num = math.floor(samples/batch_size)
        new_dataset_size = batch_num * batch_size
        #split = samples - new_dataset_size
        return int(new_dataset_size)