import os
from typing import Optional, Sequence
import numpy as np
from pytorch_lightning.core import datamodule
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


from warnings import warn


from torchvision import transforms as transform_lib
from torchvision.datasets import SVHN

from Contrastive_uncertainty.general.datamodules.dataset_normalizations import svhn_normalization
from Contrastive_uncertainty.general.datamodules.datamodule_transforms import dataset_with_indices, dataset_with_indices_SVHN



class SVHNDataModule(LightningDataModule):

    name = 'svhn'
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
        self.DATASET = SVHN
        self.DATASET_with_indices = dataset_with_indices(self.DATASET)
        #self.DATASET_with_indices = dataset_with_indices_SVHN(self.DATASET)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split


    @property
    def total_samples(self):
        """
        Return:
            630_420
        """
        return 630_420
        
    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10
    
    @property
    def num_channels(self):
        """
        Return:
            3
        """
        return 3

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.DATASET(self.data_dir, split ='train', download=True,transform=transform_lib.ToTensor())
        self.DATASET(self.data_dir, split ='test', download=True,transform=transform_lib.ToTensor())
    
    @property
    def input_height(self):
        """
        Return:
            32
        """
        return 32
        

    def setup(self):
        ''' 
        Sets up the train, val and test datasets
        '''
        self.setup_train()
        self.setup_val()
        self.setup_test()

        # Obtain class indices
        # Obtain class indices
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = self.DATASET_with_indices(self.data_dir, split ='train', download=False, transform=train_transforms, **self.extra_args)
        self.idx2class = {i:f'class {i}' for i in range(max(dataset.labels)+1)}
        #self.idx2class = {v:f'{i} - {k}'for i, (k, v) in zip(range(len(dataset.class_to_idx)),dataset.class_to_idx.items())}
        # Need to change key and value around to get in the correct order
        #self.idx2class = {k:v for k,v in self.idx2class.items() if k < self.num_classes}  

    def setup_train(self):
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = self.DATASET_with_indices(self.data_dir, split ='train', download=False, transform=train_transforms, **self.extra_args)
        

        if isinstance(dataset.labels, list):
            dataset.labels = torch.Tensor(dataset.labels).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(dataset.labels,np.ndarray):
            dataset.labels = torch.from_numpy(dataset.labels).type(torch.int64)

        dataset.targets = dataset.labels
        
        train_length = len(dataset)
        self.train_dataset, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def setup_val(self):
        
        '''
        val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=val_transforms, **self.extra_args)
        train_length = len(dataset)
        _, self.val_dataset = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        '''
        val_train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        val_train_dataset = self.DATASET_with_indices(self.data_dir, split ='train', download=False, transform=val_train_transforms, **self.extra_args)

        if isinstance(val_train_dataset.labels, list):
            val_train_dataset.labels = torch.Tensor(val_train_dataset.labels).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(val_train_dataset.labels,np.ndarray):
            val_train_dataset.labels = torch.from_numpy(val_train_dataset.labels).type(torch.int64)

        val_train_dataset.targets = val_train_dataset.labels
        train_length = len(val_train_dataset)
        _, self.val_train_dataset = random_split(
            val_train_dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        val_test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        val_test_dataset = self.DATASET_with_indices(self.data_dir, split ='train', download=False, transform=val_test_transforms, **self.extra_args)
        
        if isinstance(val_test_dataset.labels, list):
            val_test_dataset.labels = torch.Tensor(val_test_dataset.labels).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(val_test_dataset.labels,np.ndarray):
            val_test_dataset.labels = torch.from_numpy(val_test_dataset.labels).type(torch.int64)

        val_test_dataset.targets = val_test_dataset.labels


        _, self.val_test_dataset = random_split(
            val_test_dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def setup_test(self):
        test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        self.test_dataset = self.DATASET_with_indices(self.data_dir, split ='test', download=False, transform=test_transforms, **self.extra_args)
        if isinstance(self.test_dataset.labels, list):
            self.test_dataset.labels = torch.Tensor(self.test_dataset.labels).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(self.test_dataset.labels,np.ndarray):
            self.test_dataset.labels = torch.from_numpy(self.test_dataset.labels).type(torch.int64)

        self.test_dataset.targets = self.test_dataset.labels

        multi_transforms = self.default_transforms() if self.multi_transforms is None else self.multi_transforms
        self.multi_dataset = self.DATASET_with_indices(self.data_dir, split ='test', download=False, transform=multi_transforms, **self.extra_args)
        if isinstance(self.multi_dataset.labels, list):
            self.multi_dataset.labels = torch.Tensor(self.multi_dataset.labels).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(self.multi_dataset.labels,np.ndarray):
            self.multi_dataset.labels = torch.from_numpy(self.multi_dataset.labels).type(torch.int64)

        self.multi_dataset.targets = self.multi_dataset.labels

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
    
    def deterministic_train_dataloader(self): #  Makes it so that the data does not shuffle (Used for the case of the hierarchical approach)
        """
        FashionMNIST train set removes a subset to use for validation
        """

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        MNIST val set uses a subset of the training set for validation
        """
        '''
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader
        '''
        val_train_loader = DataLoader(
            self.val_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_test_loader = DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return [val_train_loader, val_test_loader]

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

    
    def multi_dataloader(self):
        """
        FashionMNIST test set uses the test split
        """

        loader = DataLoader(
            self.multi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        svhn_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            svhn_normalization()
        ])
        return svhn_transforms


