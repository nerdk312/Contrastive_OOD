SimCLR_hparams = dict(
num_samples = 45000,
loss_temperature = 0.5,


# optimizer args
lr= 1e-4,#3e-4,
opt_weight_decay = 1e-6,
warmup_epochs = 10,

bsz = 8,
dataset = 'FashionMNIST',
OOD_dataset = 'MNIST',

# Miscellaneous arguments
seed = 42,
epochs = 300,

# Trainer configurations
fast_run = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,
model_saving = 100, # Used to control how often the model is saved

project = 'SimCLR'# evaluation, Moco_training
)