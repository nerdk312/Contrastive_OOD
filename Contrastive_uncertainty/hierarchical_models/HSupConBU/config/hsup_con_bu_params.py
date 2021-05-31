hsup_con_bu_hparams = dict(
emb_dim = 128,
num_negatives = 65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

bsz = 256,
dataset = 'MNIST',
OOD_dataset = ['FashionMNIST','KMNIST'],

contrast_mode = 'one',
use_mlp = True,

# Miscellaneous arguments
seed = 42,
epochs = 300,

# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 20,
model_saving = 200, # Used to control how often the model is saved
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',
branch_weights = [1.0/3, 1.0/3, 1.0/3],
vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],
callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis'],


model_type ='HSupConBU',
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)