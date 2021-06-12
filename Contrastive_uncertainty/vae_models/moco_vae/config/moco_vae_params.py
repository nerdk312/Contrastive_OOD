moco_vae_hparams = dict(


# optimizer args
optimizer = 'adam',
learning_rate= 3e-4,
momentum= 0.9,
weight_decay = 1e-4,

bsz = 256,
dataset = 'MNIST',
OOD_dataset = ['FashionMNIST'],


# VAE specific params
emb_dim = 128,
instance_encoder = 'resnet18',
kl_coeff = 0.1,
first_conv = False,
maxpool1 = False,
enc_out_dim = 128,


# Miscellaneous arguments
seed = 42,
epochs = 300,


# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 10,
model_saving = 200, # Used to control how often the model is saved
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

#vector_level = ['instance'],
#label_level = ['fine'],
callbacks = ['Variational'],
#callbacks = ['Model_saving','MMD_instance','Metrics','Visualisation','Mahalanobis'],


model_type = 'MocoVAE',
project = 'evaluation',# evaluation, Moco_training
group = None,
notes = None,
)