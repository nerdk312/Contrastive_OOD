toy_hparams = dict(
hidden_dim = 20,
embed_dim = 2,

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,





# Miscellaneous arguments
seed = 42,
epochs = 200,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 5,

project = 'toy'# evaluation, Moco_training
)