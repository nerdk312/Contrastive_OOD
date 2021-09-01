import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split,  Dataset, Subset

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
    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([0, 0.00001])], p=0.5),
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
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class ToyMultiTwoMoonsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        # image augmentation functions
        self.multi_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([0, 0.00001])], p=0.5),
        ])

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(10)]
        return multiple_aug_inp

class ToyTrainGaussianBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,lower_bound=0.01,upper_bound=0.05):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([lower_bound, upper_bound])], p=0.5),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class ToyEvalGaussianBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k

class ToyTrainTwoGaussiansTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self,lower_bound=0.01,upper_bound=0.05):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([lower_bound, upper_bound])], p=0.5),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class ToyEvalTwoGaussiansTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianNoise(object):
    """Gaussian Noise augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.01, 0.02)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x  + (sigma*torch.randn_like(x))  # adding zero mean gaussian noise
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def Normalize(x, mean, std):
    x = (x - mean) / std
    return x



class ToyTrainBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
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


class ToyEvalBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        self.test_transform = transforms.Compose([
        ])

    def __call__(self, inp):
        q = self.test_transform(inp)
        #q = Normalize(q, mean, std)
        k = self.test_transform(inp)
        #k = Normalize(k, mean, std)
        return q, k

class ToyMultiBlobsTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """
    def __init__(self):
        # image augmentation functions
        self.multi_transform = transforms.Compose([
            transforms.RandomApply([GaussianNoise([.1, 2.])], p=0.5),
        ])

    def __call__(self, inp):
        multiple_aug_inp = [self.multi_transform(inp) for i in range(10)]
        return multiple_aug_inp



# Use to apply transforms to the tensordataset  https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        
        # y is from the 1st value to the last value to be able to deal with the coarse values which are present for the task
        

        if len(self.tensors) ==3:
            y = self.tensors[1][index]
            coarse_y = self.tensors[2][index]
            return x, y, coarse_y, index # Added the return of index for the purpose of PCL
            
        else:
            y = self.tensors[1][index]

            return x, y, index # Added the return of index for the purpose of PCL

    def __len__(self):
        return self.tensors[0].size(0)

