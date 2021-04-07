import random
import torch
from torchvision import transforms

#mean = torch.tensor([0.57647171, 0.57647171])
#std = torch.tensor([0.28364347, 0.28364347])
mean = torch.tensor([1.77806405, 1.77806405])
std = torch.tensor([1.12455573, 1.12455573])
class ToyTrainDiagonalLinesTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([.1, 2.])], p=0.5),
        ])

    def __call__(self, inp):
        
        q = self.train_transform(inp)
        #print('pre normalised q',q)
        #q = Normalize(q,mean,std)
        #print('post normalised q',q)
        k = self.train_transform(inp)
        #k = Normalize(k,mean,std)
        return q, k


class ToyEvalDiagonalLinesTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        #q = Normalize(q, mean, std)
        k = self.test_transform(inp)
        #k = Normalize(k, mean, std)
        return q, k
        

class ToyTrainTwoMoonsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        # image augmentation functions
        self.train_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class ToyEvalTwoMoonsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self, height=28):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianNoise(object):
    """Gaussian Noise augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x #  + (sigma*torch.randn_like(x))  # adding zero mean gaussian noise 
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def Normalize(x,mean,std):
    x = (x -mean)/std
    return x