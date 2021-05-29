import numpy as np
import random
from warnings import warn
from Contrastive_uncertainty.general.datamodules.dataset_normalizations import cifar10_normalization,\
    cifar100_normalization, fashionmnist_normalization, mnist_normalization, kmnist_normalization,\
    svhn_normalization, stl10_normalization, emnist_normalization
    


from PIL import ImageFilter
from torchvision import transforms


# CIFAR100 Coarse labels
CIFAR100_coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                           3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                           6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                           0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                           5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                           16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                           10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                           2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                          18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    

# MNIST Coarse labels
MNIST_coarse_labels = np.array([ 0, 2, 1,  4,  3,  4,  0,  2, 1, 3])


# FashionMNIST Coarse labels
# {0: '0 - T-shirt/top', 1: '1 - Trouser', 2: '2 - Pullover', 3: '3 - Dress', 4: '4 - Coat', 5: '5 - Sandal', 6: '6 - Shirt', 7: '7 - Sneaker', 8: '8 - Bag', 9: '9 - Ankle boot'}
# Group together [(t-shirt, shirt), (trouser), (pullover, coat), (dress), (sandal, sneaker,ankle boot),(bag)]
FashionMNIST_coarse_labels = np.array([0,1,2,3,2,4,0,4,5,4])

#KMNIST Coarse labels
# https://github.com/rois-codh/kmnist Grouping decided based on how the prototypes look like from this link
#{0: 'o', 1: 'ki', 2: 'su', 3: 'tsu', 4: 'na', 5: 'ha', 6: 'ma', 7: 'ya', 8: 're', 9: 'wo'}
# Group together [(o,tsu), (ki,ma),(su),(na,ha),(ya,re),(wo)]
KMNIST_coarse_labels = np.array([0, 1, 2, 0, 3, 3, 1, 4, 4, 5])

# CIFAR10 Coarse labels
#{0: '0 - airplane', 1: '1 - automobile', 2: '2 - bird', 3: '3 - cat', 4: '4 - deer', 5: '5 - dog', 6: '6 - frog', 7: '7 - horse', 8: '8 - ship', 9: '9 - truck'}
CIFAR10_coarse_labels = np.array([ 0,  2, 3,  5,  6,  5,  4,  6, 1,  2])



class Moco2TrainCIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalCIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            cifar10_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainCIFAR100Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar100_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalCIFAR100Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            cifar100_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainSVHNTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            svhn_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalSVHNTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=32):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            svhn_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainSTL10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=96):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            stl10_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class Moco2EvalSTL10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=96):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            stl10_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k




class Moco2TrainFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fashionmnist_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalFashionMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            fashionmnist_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            mnist_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            mnist_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainKMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            kmnist_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalKMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            kmnist_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class Moco2TrainEMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(height, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            emnist_normalization()
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalEMNISTTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
            transforms.Resize(height + 12),
            transforms.CenterCrop(height),
            transforms.ToTensor(),
            emnist_normalization(),
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k
    


        

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        
        data, target = cls.__getitem__(self, index)
        return data, target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    }) 
    '''type(name,bases,dict)
    name is the name of the class which corresponds to the __name__ attribute__
    bases: tupe of clases from which corresponds to the __bases__ attribute
    '''

'''
#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices_hierarchy(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        data, target = cls.__getitem__(self, index)
        coarse_target = coarse_labels[target]
        return data, target, coarse_target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    }) 
    
'''

#https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices_hierarchy(cls, coarse_labels_map):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        data, target = cls.__getitem__(self, index)
        coarse_target = coarse_labels_map[target]
        return data, target, coarse_target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


# Subtracts target by 1 to make it so that number of classes go from 0 to 25 rather than 1 to 26
def dataset_with_indices_emnist(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    #import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        target = target -1 #
        return data, target, index
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    }) 
    '''type(name,bases,dict)
    name is the name of the class which corresponds to the __name__ attribute__
    bases: tupe of clases from which corresponds to the __bases__ attribute
    '''


