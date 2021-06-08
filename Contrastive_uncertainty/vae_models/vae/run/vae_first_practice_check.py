from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.vae.config.vae_params import vae_hparams
from Contrastive_uncertainty.vae_models.vae.models.vae_module import VAEModule
from Contrastive_uncertainty.vae_models.vae.models.vae_model_instance import ModelInstance

vae_hparams['bsz'] = 64
vae_hparams['epochs'] = 1
vae_hparams['fast_run'] = True
vae_hparams['training_ratio'] = 0.01
vae_hparams['validation_ratio'] = 0.2
vae_hparams['test_ratio'] = 0.2
vae_hparams['val_check'] = 1
vae_hparams['project'] = 'practice'  # evaluation, Moco_training
vae_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
vae_hparams['quick_callback'] = True

train(vae_hparams,VAEModule, ModelInstance)
