from Contrastive_uncertainty.toy_example.datamodules.two_moons_datamodule import TwoMoonsDataModule
from Contrastive_uncertainty.toy_example.datamodules.straight_lines_datamodule import StraightLinesDataModule
from Contrastive_uncertainty.toy_example.datamodules.toy_transforms import ToyTrainTwoMoonsTransforms, ToyEvalTwoMoonsTransforms,\
                                                                     ToyTrainDiagonalLinesTransforms, ToyEvalDiagonalLinesTransforms
                                                                    

# Nested dict to hold the name of the dataset aswell as the different transforms for the dataset : https://www.programiz.com/python-programming/nested-dictionary

dataset_dict = {'TwoMoons':{'module':TwoMoonsDataModule,'train_transform':ToyTrainTwoMoonsTransforms(),'val_transform':ToyEvalTwoMoonsTransforms(),'test_transform':ToyEvalTwoMoonsTransforms()},
                'StraightLines': {'module':StraightLinesDataModule,'train_transform':ToyTrainDiagonalLinesTransforms(),'val_transform':ToyEvalDiagonalLinesTransforms(),'test_transform':ToyEvalDiagonalLinesTransforms()}}

