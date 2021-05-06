from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.Contrastive.config.contrastive_params import contrastive_hparams
from Contrastive_uncertainty.Contrastive.models.contrastive_module import ContrastiveModule
from Contrastive_uncertainty.Contrastive.models.contrastive_model_instance import ModelInstance

contrastive_hparams['bsz'] = 16
contrastive_hparams['instance_encoder'] = 'resnet18'
contrastive_hparams['epochs'] = 1
contrastive_hparams['fast_run'] = True
contrastive_hparams['training_ratio'] = 0.01
contrastive_hparams['validation_ratio'] = 0.2
contrastive_hparams['test_ratio'] = 0.2
contrastive_hparams['val_check'] = 1
contrastive_hparams['project'] = 'practice'  # evaluation, contrastive_training
contrastive_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
contrastive_hparams['quick_callback'] = True

train(contrastive_hparams,ContrastiveModule, ModelInstance)
