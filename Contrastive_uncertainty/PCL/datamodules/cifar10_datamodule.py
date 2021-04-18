import os
from typing import Optional, Sequence
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset


from Contrastive_uncertainty.PCL.datamodules.dataset_normalizations import cifar10_normalization
from Contrastive_uncertainty.PCL.datamodules.datamodule_transforms import dataset_with_indices,split_size
from warnings import warn


from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10



class CIFAR10DataModule(LightningDataModule):

    name = 'cifar10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
            Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
            :width: 400
            :alt: CIFAR-10
        Specs:
            - 10 classes (1 per class)
            - Each image is (3 x 32 x 32)
        Standard CIFAR10, train, val, test splits and transforms
        Transforms::
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
        Example::
            from pl_bolts.datamodules import CIFAR10DataModule
            dm = CIFAR10DataModule(PATH)
            model = LitModel()
            Trainer().fit(model, dm)
        Or you can set your own transforms
        Example::
            dm.train_transforms = ...
            dm.test_transforms = ...
            dm.val_transforms  = ...
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.DATASET_with_indices = dataset_with_indices(self.DATASET)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split

    

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.DATASET(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)

    def setup(self):
        ''' 
        Sets up the train, val and test datasets
        '''
        self.setup_train()
        self.setup_val()
        self.setup_test()

        # Obtain class indices
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        # Obtains a class where there is 
        dataset = self.DATASET_with_indices(self.data_dir, train=True, download=False, transform=train_transforms, **self.extra_args)
        self.idx2class = {v:f'{i} - {k}'for i, (k, v) in zip(range(len(dataset.class_to_idx)),dataset.class_to_idx.items())}
        # Need to change key and value around to get in the correct order
        self.idx2class = {k:v for k,v in self.idx2class.items() if k < self.num_classes}  

    def setup_train(self):
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = self.DATASET_with_indices(self.data_dir, train=True, download=False, transform=train_transforms, **self.extra_args)
        
        train_length = len(dataset)
        new_dataset_size = split_size(self.batch_size,train_length)
        indices = range(new_dataset_size)

        self.train_dataset = Subset(dataset, indices)  # Obtain a subset of the data from 0th index to the index for the last value

        '''
        self.train_dataset, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        '''
        #self.datasize =train_length - self.val_split

    

    def setup_val(self):
        # val transforms use the test transforms in this case
        val_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        dataset = self.DATASET_with_indices(self.data_dir, train=True, download=False, transform=val_transforms, **self.extra_args)

        train_length = len(dataset)
        new_dataset_size = split_size(self.batch_size,train_length)
        indices = range(new_dataset_size)
        self.val_dataset = Subset(dataset, indices) # Obtain a subset of the data from 0th index to the index for the last value
        '''
        train_length = len(dataset)
        self.val_dataset, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        '''
    def setup_test(self):
        test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        self.test_dataset = self.DATASET_with_indices(self.data_dir, train=False, download=False, transform=test_transforms, **self.extra_args)
        if isinstance(self.test_dataset.targets, list):
            self.test_dataset.targets = torch.Tensor(self.test_dataset.targets).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(self.test_dataset.targets,np.ndarray):
            self.test_dataset.targets = torch.from_numpy(self.test_dataset.targets).type(torch.int64)
        

        test_length = len(self.test_dataset)
        new_dataset_size = split_size(self.batch_size,test_length)
        indices = range(new_dataset_size)

        self.test_dataset = Subset(self.test_dataset,indices) # Obtain a subset of the data from 0th index to the index for the last value
        
        '''       
        self.test_dataset, _ = random_split(
            self.test_dataset,
            [test_length - test_split, test_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        '''

    def train_dataloader(self):
        """
        FashionMNIST train set removes a subset to use for validation
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        MNIST val set uses a subset of the training set for validation
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        #import ipdb; ipdb.set_trace()
        return loader

    def test_dataloader(self):
        """
        FashionMNIST test set uses the test split
        """

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cf10_transforms
