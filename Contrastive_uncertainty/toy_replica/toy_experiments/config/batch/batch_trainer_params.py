group1_hparams = dict(

# Miscellaneous arguments
seed = 26,
epochs = 1,
bsz = 64,

# Trainer configurations
fast_run = False,
quick_callback = True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
typicality_bootstrap = 50,
typicality_batch = 25,

callbacks =['Visualisation'],
)

group2_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 1,
bsz = 64,

# Trainer configurations
fast_run = False,
quick_callback = True,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
typicality_bootstrap = 50,
typicality_batch = 25,



callbacks =['Metrics'],

)

batch_trainer_hparams = {'Practice group 1':group1_hparams, 'Practice group 2':group2_hparams}
#import ipdb; ipdb.set_trace()
