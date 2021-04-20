evaluation_hparams = dict(
num_classes = 10,
emb_dim = 128,
num_negatives = 65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.003,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,


bsz = 256,
z_dim = 512,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',

classifier = True,
normalize = False,
contrastive = True,
use_mlp = True,


# Miscellaneous arguments
seed = 42,
epochs = 200,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 100, # Used to control how often the model is saved
pretrained_network = 'Pretrained_models/finetuned_network.pt',

project = 'evaluation'  # evaluation, Moco_training
)