import random
import math
from warnings import warn


from PIL import ImageFilter
from torchvision import transforms

from Contrastive_uncertainty.general_clustering.datamodules.dataset_normalizations import  cifar10_normalization, fashionmnist_normalization,mnist_normalization,svhn_normalization

from Contrastive_uncertainty.general.datamodules.datamodule_transforms import \
    Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms, \
    Moco2TrainSVHNTransforms, Moco2EvalSVHNTransforms,\
    Moco2TrainFashionMNISTTransforms, Moco2EvalFashionMNISTTransforms, \
    Moco2TrainMNISTTransforms, Moco2EvalMNISTTransforms, \
    GaussianBlur, dataset_with_indices



def split_size(batch_size, samples): # obtains a dataset size for the k-means based on the batch size
        batch_num = math.floor(samples/batch_size)
        new_dataset_size = batch_num * batch_size
        #split = samples - new_dataset_size
        return int(new_dataset_size)