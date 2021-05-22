from Contrastive_uncertainty.toy_example.datamodules.two_moons_datamodule import TwoMoonsDataModule
from Contrastive_uncertainty.toy_example.datamodules.two_moons_hierarchy_datamodule import TwoMoonsHierarchyDataModule
from Contrastive_uncertainty.toy_example.datamodules.blobs_datamodule import BlobsDataModule
from Contrastive_uncertainty.toy_example.datamodules.straight_lines_datamodule import StraightLinesDataModule
from Contrastive_uncertainty.toy_example.datamodules.toy_transforms import ToyTrainTwoMoonsTransforms, ToyEvalTwoMoonsTransforms,\
                                                                     ToyTrainDiagonalLinesTransforms, ToyEvalDiagonalLinesTransforms, \
                                                                         ToyTrainBlobsTransforms, ToyEvalBlobsTransforms
                                                                    

# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

dataset_dict = {'TwoMoons':{'module':TwoMoonsDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},
                'TwoMoonsHierarchy':{'module':TwoMoonsHierarchyDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},
                'Blobs':{'module':BlobsDataModule,'train_transform':ToyTrainBlobsTransforms(),'val_transform':ToyEvalBlobsTransforms(),'test_transform':ToyEvalBlobsTransforms()},
                'StraightLines': {'module':StraightLinesDataModule,'train_transform':ToyTrainDiagonalLinesTransforms(),'val_transform':ToyEvalDiagonalLinesTransforms(),'test_transform':ToyEvalDiagonalLinesTransforms()}}

