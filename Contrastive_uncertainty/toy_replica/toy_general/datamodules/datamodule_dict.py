from Contrastive_uncertainty.toy_replica.toy_general.datamodules.two_moons_datamodule import TwoMoonsDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.blobs_datamodule import BlobsDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.diagonal_lines_datamodule import DiagonalLinesDataModule
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.toy_transforms import ToyTrainTwoMoonsTransforms, ToyEvalTwoMoonsTransforms,ToyMultiTwoMoonsTransforms,\
                                                                     ToyTrainDiagonalLinesTransforms, ToyEvalDiagonalLinesTransforms, \
                                                                        ToyTrainBlobsTransforms, ToyEvalBlobsTransforms, ToyMultiBlobsTransforms
                                                                    

# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

# Changed the multi_transform to not use () as I will provide it an input which controls the number of data augmentations used
dataset_dict = {'TwoMoons':{'module':TwoMoonsDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms(),
                           'multi_transform':ToyMultiTwoMoonsTransforms},
                
                'Blobs':{'module':BlobsDataModule,'train_transform':ToyTrainBlobsTransforms(),'val_transform':ToyEvalBlobsTransforms(),'test_transform':ToyEvalBlobsTransforms(),
                        'multi_transform':ToyMultiBlobsTransforms}}
'''
dataset_dict = {'TwoMoons':{'module':TwoMoonsDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},
                
                'Blobs':{'module':BlobsDataModule,'train_transform':ToyTrainBlobsTransforms(),'val_transform':ToyEvalBlobsTransforms(),'test_transform':ToyEvalBlobsTransforms()},
                
                'Diagonal':{'module':DiagonalLinesDataModule,'train_transform':ToyTrainDiagonalLinesTransforms(),'val_transform':ToyEvalDiagonalLinesTransforms(),'test_transform':ToyEvalDiagonalLinesTransforms()}}

'''
#'StraightLines': {'module':StraightLinesDataModule,'train_transform':ToyTrainDiagonalLinesTransforms(),'val_transform':ToyEvalDiagonalLinesTransforms(),'test_transform':ToyEvalDiagonalLinesTransforms()}
#'TwoMoonsHierarchy':{'module':TwoMoonsHierarchyDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},




OOD_dict = {'TwoMoons':['Blobs','Diagonal'],
            'Blobs':['TwoMoons','Diagonal'],
            'Diagonal':['Blobs','TwoMoons']}



