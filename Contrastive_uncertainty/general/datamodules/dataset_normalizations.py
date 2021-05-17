from warnings import warn

from torchvision import transforms

def imagenet_normalization():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize

def cifar10_normalization():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    return normalize
#https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 - Normalisation obtained from here
def cifar100_normalization():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                                    
    return normalize

def stl10_normalization():
    normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize

def fashionmnist_normalization():
    normalize = transforms.Normalize(mean =(0.2861,),std =(0.3530,))
    return normalize

def mnist_normalization():
    normalize = transforms.Normalize(mean =(0.1307,),std =(0.3081,))
    return normalize

def kmnist_normalization():
    normalize = transforms.Normalize(mean =(0.1918,),std =(0.3385,))
    return normalize

def emnist_normalization():
    normalize = transforms.Normalize(mean =(0.1344,),std =(1.0520,))
    return normalize

def svhn_normalization():
    normalize = transforms.Normalize(mean = [0.4380,0.440,0.4730], std = [0.1751,0.1771,0.1744])
    return normalize
