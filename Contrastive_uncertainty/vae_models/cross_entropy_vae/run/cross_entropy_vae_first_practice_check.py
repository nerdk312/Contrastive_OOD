from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.cross_entropy_vae.config.cross_entropy_vae_params import cross_entropy_vae_hparams
from Contrastive_uncertainty.vae_models.cross_entropy_vae.models.cross_entropy_vae_module import CrossEntropyVAEModule

from Contrastive_uncertainty.vae_models.cross_entropy_vae.models.cross_entropy_vae_model_instance import ModelInstance


cross_entropy_vae_hparams['bsz'] = 16
#vae_hparams['emb_dim'] = 32
#vae_hparams['enc_out_dim'] = 32
cross_entropy_vae_hparams['epochs'] = 1
cross_entropy_vae_hparams['fast_run'] = True
cross_entropy_vae_hparams['training_ratio'] = 0.01
cross_entropy_vae_hparams['validation_ratio'] = 0.2
cross_entropy_vae_hparams['test_ratio'] = 0.2
cross_entropy_vae_hparams['val_check'] = 1
cross_entropy_vae_hparams['project'] = 'practice'  # evaluation, Moco_training
cross_entropy_vae_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
cross_entropy_vae_hparams['quick_callback'] = True

train(cross_entropy_vae_hparams, CrossEntropyVAEModule, ModelInstance)
