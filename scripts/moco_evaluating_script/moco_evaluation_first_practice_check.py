from evaluation.evaluate_moco import eval_moco
from scripts.config.evaluation_params import evaluation_hparams

evaluation_hparams['bsz'] =16
evaluation_hparams['instance_encoder'] ='resnet18'
evaluation_hparams['epochs'] = 1
evaluation_hparams['fast_run'] = True
evaluation_hparams['training_ratio'] = 0.01
evaluation_hparams['validation_ratio'] = 0.2
evaluation_hparams['test_ratio'] = 0.2
evaluation_hparams['val_check'] = 1
evaluation_hparams['project'] = 'practice' # evaluation, Moco_training
evaluation_hparams['pretrained_network'] = None
evaluation_hparams['quick_callback'] = True
eval_moco(evaluation_hparams)