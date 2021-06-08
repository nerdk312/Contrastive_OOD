from Contrastive_uncertainty.general.train.train_general import train
from Contrastive_uncertainty.vae_models.sup_con_vae.config.sup_con_vae_params import sup_con_vae_hparams
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_module import SupConVAEModule
from Contrastive_uncertainty.vae_models.sup_con_vae.models.sup_con_vae_model_instance import ModelInstance

sup_con_vae_hparams['bsz'] = 16
#vae_hparams['emb_dim'] = 32
#vae_hparams['enc_out_dim'] = 32
sup_con_vae_hparams['epochs'] = 1
sup_con_vae_hparams['fast_run'] = True
sup_con_vae_hparams['training_ratio'] = 0.01
sup_con_vae_hparams['validation_ratio'] = 0.2
sup_con_vae_hparams['test_ratio'] = 0.2
sup_con_vae_hparams['val_check'] = 1
sup_con_vae_hparams['project'] = 'practice'  # evaluation, Moco_training
sup_con_vae_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
sup_con_vae_hparams['quick_callback'] = True

train(sup_con_vae_hparams, SupConVAEModule, ModelInstance)
