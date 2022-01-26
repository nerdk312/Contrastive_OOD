
sup_con_hparams = dict(
emb_dim = 128,
softmax_temperature = 0.07,
contrast_mode = 'one',
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'MNIST',
OOD_dataset = ['FashionMNIST'],

model_type = 'SupCon',
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,


bsz = 256,
# Miscellaneous arguments
seed = 42,
epochs = 300,


num_augmentations = 5, # Used to control the number of data augmentations for multiloader callback


# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved

callbacks = ['Model_saving'],
)
