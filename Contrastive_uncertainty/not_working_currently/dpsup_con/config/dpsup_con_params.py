dpsup_con_hparams = dict(
num_classes = 10,
emb_dim = 128,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,
distance_threshold = 1.0,

bsz = 256,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',

use_mlp = True,


# Miscellaneous arguments
seed = 42,
epochs = 400,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',


project = 'evaluation'# evaluation, Moco_training
)