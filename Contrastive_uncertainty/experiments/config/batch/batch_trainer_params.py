group1_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 64,
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
typicality_bootstrap = 50,
typicality_batch = 25,

# Used to update callback dict
vector_level = ['fine'],
label_level =['coarse'],
#vector_level = ['fine'],
#label_level = ['coarse'],

callbacks = ['General Scores','Visualisation'],
)


group2_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 64,
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
typicality_bootstrap = 50,
typicality_batch = 25,

# Used to update callback dict
vector_level = ['fine'],
label_level =['coarse'],


callbacks = ['General Scores','Visualisation'],
)


batch_trainer_hparams = {'Practice group 1':group1_hparams, 'Practice group 2':group2_hparams}
