base_hparams = dict(
# Optimizer parameters in common
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

# Training parameters in common
emb_dim = 128,
instance_encoder = 'resnet50',
bsz = 256,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',
pretrained_network = None,

# Miscellaneous arguments in common
seed = 26,
epochs = 300,

# Trainer configurations in common
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

# Wandb parameters in common
project = 'evaluation',
group = None,
notes = None, # Add notes to the specific models each time


# Cross entropy Specific parameters
num_classes = 10,
label_smoothing = False,

# Contrastive specific parameters
Contrastive = True,
supervised_contrastive = False,
num_negatives = 4096,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
use_mlp =True,

# Supcon specific parameters
contrast_mode ='one',

# PCL specific parameters
num_multi_cluster = [5000,10000],
num_inference_cluster = [10,100,1000], # Number of clusters for the inference callback


# unsupcon memory parameters
memory_momentum = 0.5,
num_cluster = [10],

# Either goes through all the models or goes through baselines

single_model = 'baselines'
)  # evaluation