imp_hparams = dict(
# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,


emb_dim = 128,
alpha = 1.0,
sigma = 1.0,
use_mlp = True,

instance_encoder = 'resnet50',
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

bsz = 256,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',

# Miscellaneous arguments
seed = 26,
epochs = 300,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

num_classes = 10,


project = 'evaluation')  # evaluation