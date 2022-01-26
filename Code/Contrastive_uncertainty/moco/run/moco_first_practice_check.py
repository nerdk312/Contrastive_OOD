
from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.moco.config.moco_params import moco_hparams
from Contrastive_uncertainty.moco.models.moco_module import MocoModule
from Contrastive_uncertainty.moco.models.moco_model_instance import ModelInstance




moco_hparams['bsz'] = 16
moco_hparams['instance_encoder'] = 'resnet18'
moco_hparams['epochs'] = 1
moco_hparams['fast_run'] = True
moco_hparams['training_ratio'] = 0.01
moco_hparams['validation_ratio'] = 0.2
moco_hparams['test_ratio'] = 0.2
moco_hparams['val_check'] = 1
moco_hparams['project'] = 'practice'  # evaluation, contrastive_training
moco_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
moco_hparams['quick_callback'] = True

train(moco_hparams, MocoModule, ModelInstance)
