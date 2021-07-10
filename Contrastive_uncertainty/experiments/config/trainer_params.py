trainer_hparams = dict(
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


#callbacks = ['Dataset_distances','classification'],
#callbacks = ['Subsample'],
#callbacks = ['Oracle'],
#callbacks = ['Hierarchical Scores','General Scores'],
#callbacks = ['Relative'],
#callbacks = ['Hierarchical Relative'],
#callbacks = ['Practice Hierarchical scores'],
#callbacks = ['Mahalanobis'],
#callbacks = ['Typicality General Point Updated'],
#callbacks = ['Hierarchical_Random_Coarse'],
#callbacks = ['Class Mahalanobis'],
callbacks = ['Hierarchical Subclusters'],
#callbacks = ['Oracle Hierarchical Metrics'],
#callbacks = ['General Scores'],
#callbacks = ['Visualisation','Metrics','Mahalanobis Distance', 'Hierarchical Mahalanobis', 'Hierarchical Scores','Oracle Hierarchical', 'General Scores'],
#callbacks = ['Typicality_OVR'],
#callbacks =['Variational'],
)