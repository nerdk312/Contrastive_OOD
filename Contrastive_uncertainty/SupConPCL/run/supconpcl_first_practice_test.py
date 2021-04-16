from Contrastive_uncertainty.SupConPCL.train.train_supconpcl import training
from Contrastive_uncertainty.SupConPCL.config.supconpcl_params  import supconpcl_hparams 


supconpcl_hparams['bsz'] = 64
supconpcl_hparams['num_cluster'] = [10,20]
supconpcl_hparams['instance_encoder'] = 'resnet18'
supconpcl_hparams['epochs'] = 1
supconpcl_hparams['fast_run'] = False
supconpcl_hparams['training_ratio'] = 0.01
supconpcl_hparams['validation_ratio'] = 0.2
supconpcl_hparams['test_ratio'] = 0.2
supconpcl_hparams['val_check'] = 1
supconpcl_hparams['project'] = 'practice'  # evaluation, Moco_training
supconpcl_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
supconpcl_hparams['quick_callback'] = True
training(supconpcl_hparams)
