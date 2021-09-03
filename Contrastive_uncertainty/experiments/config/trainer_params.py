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
typicality_batch = 10,
num_augmentations = 5,

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
#callbacks = ['Hierarchical Subclusters'],
#callbacks = ['IForest'],
#callbacks = ['Class Variance'],
#callbacks = ['Class Radii'],

#callbacks = ['Class Variance'],
#callbacks = ['One Dimensional Background Mahalanobis'],
#callbacks = ['Class Relative Mahalanobis', 'Class Inverted Relative Mahalanobis'],
#callbacks = ['One Dimensional Relative Mahalanobis Variance','One Dimensional Mahalanobis Variance'],
#callbacks = ['One Dimensional Mahalanobis Similarity'],

#callbacks = ['Marginal Typicality OOD'],
#callbacks = ['Centroid Relative Distances'],
#callbacks = ['Confusion Log Probability'],
#callbacks = ['Visualisation','Metrics'],
#callbacks = ['Marginal Typicality Entropy Mean','Mahalanobis Distance','Class Mahalanobis'],
#callbacks = ['Feature Entropy', 'Centroid_distances'],#,'Total Centroid KL'],
#callbacks = ['Total Centroid KL'],

#callbacks = ['Centroid Distances','MMD'],
#callbacks = ['Point One Dim Class Typicality Normalised'],
#callbacks = ['Point One Dim Relative Class Typicality Normalised'],
callbacks = ['Data Augmented Point One Dim Class Typicality Normalised'],

#callbacks = ['Alternative Data Augmented Point One Dim Class Typicality Normalised','Data Augmented Mahalanobis'],


#callbacks = ['One Dim Typicality'],
#callbacks = ['One Dim Typicality Class'],
#callbacks = ['One Dim Typicality Marginal Oracle'],
#callbacks = ['One Dim Typicality Marginal Batch', 'One Dim Typicality Normalised Marginal Batch'],
#callbacks = ['Marginal Typicality Entropy Mean'],
#callbacks = ['Total Centroid KL','Class Centroid Radii Overlap'],
#callbacks = ['Total Centroid KL'],
#callbacks = ['Class Centroid Radii Overlap'],

#callbacks = ['Typicality_OVR_diff_bsz_updated'],
#callbacks = ['Typicality_OVR'],
#callbacks = ['Feature Entropy'],
#callbacks = ['Class One Dimensional Mahalanobis OOD Similarity'],

#callbacks = ['Class One Dimensional Mahalanobis', 'Class One Dimensional Relative Mahalanobis'],
#callbacks = ['Class Radii Histograms'],
#callbacks = ['Oracle Hierarchical Metrics'],
#callbacks = ['General Scores'],
#callbacks = ['Visualisation','Metrics','Mahalanobis Distance', 'Hierarchical Mahalanobis', 'Hierarchical Scores','Oracle Hierarchical', 'General Scores'],
#callbacks = ['Typicality_OVR'],
#callbacks =['Variational'],
)