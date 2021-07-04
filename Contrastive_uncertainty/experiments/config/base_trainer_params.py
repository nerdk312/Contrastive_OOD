from pytorch_lightning import callbacks
from Contrastive_uncertainty.general.datamodules.datamodule_dict import OOD_dict


practice_hparams = dict(
# Optimizer parameters in common
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet18',
bsz = 64,
dataset = 'MNIST',
#OOD_dataset = ['SVHN','CIFAR10'],
pretrained_network = None,

# Miscellaneous arguments in common
seed = 26,
epochs = 300,

# Trainer configurations in common
fast_run = True,
quick_callback = True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved
typicality_bootstrap = 50,
typicality_batch = 25,

# Wandb parameters in common
project = 'practice',
group = None,
notes = None, # Add notes to the specific models each time



# VAE specific params
kl_coeff = 0.1,
first_conv = False,
maxpool1 = False,
enc_out_dim = 128,


# Cross entropy Specific parameters

label_smoothing = False,

# Contrastive specific parameters
num_negatives = 128,
encoder_momentum = 0.999,
softmax_temperature = 0.07,


# Supcon specific parameters
contrast_mode ='one',

# PCL specific parameters
num_multi_cluster = [100,500],
num_cluster_negatives = 4096,
num_inference_cluster = [10,100,1000], # Number of clusters for the inference callback


# unsupcon memory parameters
memory_momentum = 0.5,
num_cluster = [10],

# HSupConBU parameters
branch_weights = [1.0/3, 1.0/3, 1.0/3],
vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],
#callbacks = ['OOD_Dataset_distances'],
#callbacks = ['Model_saving','Variational'],
#callbacks = ['Aggregated','Differing'],
#callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis','Dataset_distances'],
#callbacks = ['Hierarchical'],
#callbacks = ['Typicality'],
#callbacks = ['Subsample'],
#callbacks = ['Practice'],
callbacks = ['General Scores'],
#callbacks = ['Practice Hierarchical scores'],
# Either goes through all the models or goes through baselines

single_model = 'Baselines'
)  # evaluation


if 'OOD_dataset' in practice_hparams:
    pass    
else:
    practice_hparams['OOD_dataset'] = OOD_dict[practice_hparams['dataset']]



trainer_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 64,
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
typicality_bootstrap = 50,
typicality_batch = 25,

# Used to update callback dict
#vector_level = ['instance', 'fine', 'coarse'],
#label_level = ['fine','fine','coarse'],
vector_level = ['fine'],
label_level =['coarse'],
#vector_level = ['fine'],
#label_level = ['coarse'],



#callbacks = ['Dataset_distances','classification'],
#callbacks = ['Subsample'],
#callbacks = ['Oracle'],
#callbacks = ['Hierarchical Scores','General Scores'],
#callbacks = ['Relative'],
#callbacks = ['Hierarchical Relative'],
#callbacks = ['Practice Hierarchical scores'],
#callbacks = ['Mahalanobis'],
callbacks = ['General Scores','Visualisation'],
#callbacks = ['Visualisation','Metrics','Mahalanobis Distance', 'Hierarchical Mahalanobis', 'Hierarchical Scores','Oracle Hierarchical', 'General Scores'],
#callbacks = ['Typicality_OVR'],
#callbacks =['Variational'],
)