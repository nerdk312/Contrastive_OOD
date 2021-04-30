pcl_hparams = dict(
# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,


emb_dim = 128,
num_negatives = 4096,#65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
num_multi_cluster = [5000,10000],
use_mlp = True,

instance_encoder = 'resnet50',
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

bsz = 256,
dataset = 'CIFAR10',
OOD_dataset = 'SVHN',

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
num_inference_cluster = [10,100,1000], # Number of clusters for the inference callback


project = 'evaluation',
group = None,
notes = None,)  # evaluation
