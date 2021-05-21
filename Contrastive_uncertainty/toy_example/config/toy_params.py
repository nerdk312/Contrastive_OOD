toy_hparams = dict(
hidden_dim = 20,
emb_dim = 2,
num_negatives = 1024,
encoder_momentum = 0.0,#0.999,
softmax_temperature = 1.0,#0.07,
num_classes = 2,
num_cluster = [10],


# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,
momentum= 0.9,
weight_decay = 1e-4,


# Miscellaneous arguments
seed = 26,
epochs = 2,
bsz = 128,
dataset = 'TwoMoons',
OOD_dataset = 'StraightLines',
model = 'HPCLSequential',

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
pretrained_network = None,

project = 'toy'  # evaluation, Moco_training
)