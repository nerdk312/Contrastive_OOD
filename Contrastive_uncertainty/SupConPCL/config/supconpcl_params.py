supconpcl_hparams = dict(
# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

num_classes = 10,
emb_dim = 128,
num_negatives = 16384,#65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
num_cluster = [10,20],
use_mlp = True,
instance_encoder = 'resnet50',
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

bsz = 512,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',

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

project = 'evaluation')  # evaluation