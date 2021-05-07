unsup_con_memory_hparams = dict(
num_classes = 10,
emb_dim = 128,
num_negatives = 8192,
memory_momentum = 0.5,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

bsz = 256,
dataset = 'KMNIST',
OOD_dataset = 'MNIST',

use_mlp = True,

# Miscellaneous arguments
seed = 42,
epochs = 500,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 10,
model_saving = 200, # Used to control how often the model is saved

num_cluster = [10],
num_inference_cluster = [10,100,1000], # Number of clusters for the inference callback

model_type ='UnSupConMemory',
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)