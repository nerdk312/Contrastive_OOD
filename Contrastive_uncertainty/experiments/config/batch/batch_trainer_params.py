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
vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],
#vector_level = ['coarse'],
#label_level =['fine'],
#vector_level = ['fine'],
#label_level = ['coarse'],

callbacks = ['Hierarchical_Random_Coarse'],


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
vector_level = ['instance', 'fine', 'coarse'],
label_level = ['fine','fine','coarse'],
#vector_level = ['coarse'],
#label_level =['fine'],
#vector_level = ['fine'],
#label_level = ['coarse'],

callbacks = ['Hierarchical_Random_Coarse'],

)


batch_trainer_hparams = {'OOD hierarchy baselines':group1_hparams, 'OOD detection at different scales experiment':group2_hparams}
