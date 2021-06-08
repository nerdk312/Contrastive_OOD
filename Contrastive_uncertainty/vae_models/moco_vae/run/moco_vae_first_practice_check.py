from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.moco_vae.config.moco_vae_params import moco_vae_hparams
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_module import MocoVAEModule
from Contrastive_uncertainty.vae_models.moco_vae.models.moco_vae_model_instance import ModelInstance

moco_vae_hparams['bsz'] = 16
#vae_hparams['emb_dim'] = 32
#vae_hparams['enc_out_dim'] = 32
moco_vae_hparams['epochs'] = 1
moco_vae_hparams['fast_run'] = True
moco_vae_hparams['training_ratio'] = 0.01
moco_vae_hparams['validation_ratio'] = 0.2
moco_vae_hparams['test_ratio'] = 0.2
moco_vae_hparams['val_check'] = 1
moco_vae_hparams['project'] = 'practice'  # evaluation, Moco_training
moco_vae_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
moco_vae_hparams['quick_callback'] = True

train(moco_vae_hparams, MocoVAEModule, ModelInstance)
