import random
import torch
from torchvision import transforms
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
        k = self.train_transform(inp)
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
        k = self.test_transform(inp)
        return q, k
        

class GaussianNoise(object):
    """Gaussian Noise augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x + (sigma*torch.randn_like(x))  # adding zero mean gaussian noise 
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x