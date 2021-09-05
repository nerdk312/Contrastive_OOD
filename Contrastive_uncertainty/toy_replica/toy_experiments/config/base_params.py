from pytorch_lightning import callbacks
from Contrastive_uncertainty.toy_replica.toy_general.datamodules.datamodule_dict import OOD_dict

base_hparams = dict(
# Optimizer parameters in common
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet50',
bsz = 64,
dataset = 'Blobs',
OOD_dataset = ['TwoMoons'],
# OOD_dataset = ['TwoMoons'],
pretrained_network = None,

# Miscellaneous arguments in common
seed = 26,
epochs = 1,

# Trainer configurations in common
fast_run = False,
quick_callback = False,#True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 1, # Used to control how often the model is saved
typicality_bootstrap = 50,
typicality_batch = 5,
num_augmentations = 2, # Used to control the number of data augmentations for multiloader callback

# Wandb parameters in common
project = 'Toy_evaluation',
group = 'Practice040721',
notes = 'Examining how to automate selection of runs',  # Add notes to the specific models each time

# Cross entropy Specific parameters
label_smoothing = False,

# Contrastive specific parameters
num_negatives = 4096,
encoder_momentum = 0.999,
softmax_temperature = 0.07,

# Supcon specific parameters
contrast_mode ='one',

# PCL specific parameters
num_multi_cluster = [2000,4000], 
num_cluster_negatives = 1024,
num_inference_cluster = [10,100,1000], # Number of clusters for the inference callback

# unsupcon memory parameters
memory_momentum = 0.5,
num_cluster = [100],

# HSupConBU parameters
branch_weights = [1.0/3, 1.0/3, 1.0/3],

vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],

#Moco divergence parameters
weighting = 0.25,

#vector_level = 'fine',
#label_level = 'coarse',
#callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis','Dataset_distances'],
#callbacks = ['Typicality_OVR_diff'],
#callbacks = ['Typicality General Point Updated'],
#callbacks = ['Class Mahalanobis'],
#callbacks = ['Oracle Hierarchical Metrics'],
#callbacks = ['Hierarchical_Random_Coarse'],
#callbacks = ['Hierarchical Subclusters'],
#callbacks = ['Class Variance'],
#callbacks = ['One Dimensional Shared Mahalanobis', 'One Dimensional Shared Relative Mahalanobis'],
#callbacks = ['Class One Dimensional Mahalanobis', 'Class One Dimensional Relative Mahalanobis'],
#callbacks = ['One Dimensional Background Mahalanobis'],
#callbacks = ['Class Relative Mahalanobis', 'Class Inverted Relative Mahalanobis'],
#callbacks = ['One Dimensional Relative Mahalanobis Variance','One Dimensional Mahalanobis Variance'],
#callbacks = ['Class One Dimensional Relative Mahalanobis Variance'],
#callbacks = ['One Dimensional Mahalanobis Similarity'],
#callbacks = ['Class One Dimensional Mahalanobis OOD Similarity'],

#callbacks = ['Marginal Typicality OOD'],
#callbacks = [],
#callbacks = ['One Dim Typicality'],
#callbacks = ['One Dim Typicality Class'],
#callbacks = ['One Dim Typicality Marginal Oracle'],
#callbacks = ['One Dim Typicality Marginal Batch'],
#callbacks = ['One Dim Typicality Normalised Marginal Batch'],
#callbacks = ['Marginal Typicality Entropy Mean'],

#callbacks = ['Total Centroid KL'],
#callbacks = ['Class Centroid Radii Overlap'],
#callbacks = ['Total Centroid KL','Class Centroid Radii Overlap'],

#callbacks = ['Feature Entropy'],
#callbacks = ['Confusion Log Probability','Model_saving'],
#callbacks = ['Bottom K Mahalanobis Difference'],
#callbacks = ['Centroid Relative Distances'],
#callbacks = ['Typicality_OVR_diff_bsz_updated'],
#callbacks = ['Typicality_OVR'],


#callbacks = ['Point One Dim Class Typicality Normalised'],
#callbacks = ['Point One Dim Relative Class Typicality Normalised'],

#callbacks = ['Data Augmented Point One Dim Class Typicality Normalised'],
#callbacks = ['Alternative Data Augmented Point One Dim Class Typicality Normalised'],
#callbacks = ['Data Augmented Mahalanobis'],
#callbacks = ['Alternative Data Augmented Point One Dim Class Typicality Normalised','Data Augmented Mahalanobis'],

#callbacks = ['Data Augmented Point One Dim Class Typicality Normalised'],
#callbacks = ['Data Augmented Alternative Point One Dim Class Typicality Normalised'],
#callbacks = ['Data Augmented Point One Dim Class Typicality Normalised','Data Augmented Alternative Point One Dim Class Typicality Normalised'],

#callbacks = ['Data Augmented Point One Dim Marginal Typicality Normalised Variance Analysis'],
#callbacks = ['Data Augmented Point One_Dim Marginal Typicality Normalised Single Variance Analysis'],
callbacks = ['Data Augmented Point One Dim Marginal Typicality Normalised Variance Analysis','Data Augmented Point One_Dim Marginal Typicality Normalised Single Variance Analysis'],

#callbacks = ['Point One Dim Relative Class Typicality Analysis'],
#callbacks = [],



#callbacks = ['Model_saving'],
#callbacks = ['Confusion Log Probability'],
#callbacks = ['Class Inverted Relative Mahalanobis'],
#callbacks = ['One Dimensional Relative Mahalanobis'],
#callbacks = ['Class Radii'],
#callbacks = ['Centroid Distances'],
#callbacks = ['Class Radii Histograms'],

#callbacks = ['Visualisation','Metrics','Mahalanobis Distance', 'Hierarchical Mahalanobis', 'Hierarchical Scores','Oracle Hierarchical', 'General Scores'],

#callbacks = ['General Scores','Visualisation'],
#callbacks = ['Dataset_distances'],
#callbacks = ['Hierarchical Scores','Subsample','Datasets'],
#callbacks = ['Subsample'],
#callbacks = ['Hierarchical Scores'],
#callbacks = ['Mahalanobis'],
#callbacks = ['Oracle'],
#callbacks = ['Practice Hierarchical scores'],
#callbacks = ['Hierarchical Scores','General Scores'],
#callbacks = ['One'],
#callbacks = ['Relative'],
#callbacks = ['Hierarchical Relative'],
#callbacks = ['Typicality_OVR','Visualisation','Mahalanobis Distance','Metrics'],
#callbacks = ['Hierarchical Scores'],
#callbacks = [],
# Either goes through all the models or goes through baselines

single_model = 'Baselines'
)  # evaluation
# Updates OOD dataset if not manually specified
if 'OOD_dataset' in base_hparams:
    pass    
else:
    base_hparams['OOD_dataset'] = OOD_dict[base_hparams['dataset']]
