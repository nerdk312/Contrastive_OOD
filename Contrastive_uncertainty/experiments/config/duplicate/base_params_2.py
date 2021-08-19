from pytorch_lightning import callbacks
from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict

base_hparams = dict(
# Optimizer parameters in common
#optimizer = 'adam', #'adam',
#learning_rate= 3e-4, #3e-4,

optimizer = 'sgd', #'adam',
learning_rate= 3e-2, #3e-4,

momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet50', # Use resnet 18 for confusion log probability 
bsz = 256,
dataset = 'CIFAR100',
#OOD_dataset = ['CIFAR10'],
#dataset = 'CIFAR100',
#OOD_dataset = ['SVHN'],
pretrained_network = None,

# Miscellaneous arguments in common
seed = 42,
epochs = 300,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved
typicality_bootstrap = 50,
typicality_batch = 25,

# Wandb parameters in common
project = 'evaluation',

group = 'Baselines Repeats',
notes = 'Repeating the hierarchical baselines',  # Add notes to the specific models each time

#group = 'Separate branch combinations',
#notes = 'Training different combinations of branches weights for the hierarchical network',  # Add notes to the specific models each time

# VAE specific params
kl_coeff = 0.1,
first_conv = False,
maxpool1 = False,
enc_out_dim = 128,


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
#branch_weights = [1.0/3, 1.0/3, 1.0/3],
branch_weights = [0.15, 0.30, 0.55],
# Either goes through all the models or goes through baselines
vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],
#callbacks = ['Model_saving','Variational'],
callbacks = ['Model_saving'],
#callbacks = ['Model_saving', 'Confusion Log Probability'],
#callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis','Dataset_distances'],
#callbacks = ['Aggregated','Differing'],

single_model = 'Baselines'
)  # evaluation

# Updates OOD dataset if not manually specified
if 'OOD_dataset' in base_hparams:
    pass    
else:
    base_hparams['OOD_dataset'] = OOD_dict[base_hparams['dataset']]
