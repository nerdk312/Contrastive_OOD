from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.cross_entropy.config.cross_entropy_params import cross_entropy_hparams
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_module import CrossEntropyModule
from Contrastive_uncertainty.cross_entropy.models.cross_entropy_model_instance import ModelInstance

cross_entropy_hparams['bsz'] = 128
cross_entropy_hparams['instance_encoder'] = 'resnet18'
cross_entropy_hparams['epochs'] = 1
cross_entropy_hparams['fast_run'] = True
cross_entropy_hparams['training_ratio'] = 0.01
cross_entropy_hparams['validation_ratio'] = 0.2
cross_entropy_hparams['test_ratio'] = 0.2
cross_entropy_hparams['val_check'] = 1
cross_entropy_hparams['project'] = 'practice'  # evaluation, Moco_training
cross_entropy_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
cross_entropy_hparams['quick_callback'] = True

train(cross_entropy_hparams,CrossEntropyModule, ModelInstance)
