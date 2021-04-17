toy_hparams = dict(
hidden_dim = 20,
emb_dim = 2,
num_negatives = 1024,
memory_momentum = 0.5,#0.999,
softmax_temperature = 1.0,#0.07,
num_classes = 2,


# optimizer args
optimizer = 'sgd',
learning_rate= 0.003,
momentum= 0.9,
weight_decay = 1e-4,


# Miscellaneous arguments
seed = 26,
epochs = 25,
bsz = 512,
dataset = 'TwoMoons',
OOD_dataset = 'StraightLines',
model = 'NCA',

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