toy_hparams = dict(
hidden_dim = 128,
emb_dim = 128,
num_negatives = 256,
encoder_momentum = 0.999,
softmax_temperature = 1.0,#0.07,
num_classes = 4,

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,
momentum= 0.9,
weight_decay = 1e-4,


# Miscellaneous arguments
seed = 42,
epochs = 200,
bsz = 256,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 5,
pretrained_network = None,

project = 'toy'# evaluation, Moco_training
)