sup_con_hparams = dict(
emb_dim = 128,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

bsz = 256,
dataset = 'Blobs',
OOD_dataset = ['TwoMoons','Diagonal'],

contrast_mode = 'one',


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
typicality_bootstrap = 50,
typicality_batch = 25,
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

#vector_level = ['instance'],
#label_level = ['fine'],
callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis'],

model_type ='SupCon',
project = 'toy_replica',# evaluation, Moco_training
group = None,
notes = None,
)