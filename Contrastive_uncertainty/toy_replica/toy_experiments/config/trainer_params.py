trainer_hparams = dict(

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


callbacks = ['Confusion Log Probability'],
#callbacks =['Visualisation'],
# Updating for the test run
#OOD_dataset = ['TwoMoons','Diagonal'],
#callbacks =['Mahalanobis'],
#callbacks = ['Dataset_distances'],
)