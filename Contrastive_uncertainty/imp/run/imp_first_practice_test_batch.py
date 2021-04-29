from Contrastive_uncertainty.imp.train.train_imp import training
from Contrastive_uncertainty.imp.config.imp_params import imp_hparams


imp_hparams['bsz'] = 128
imp_hparams['num_cluster'] = [5000]
imp_hparams['instance_encoder'] = 'resnet18'
imp_hparams['epochs'] = 1
imp_hparams['fast_run'] = True 
imp_hparams['training_ratio'] = 0.01
imp_hparams['validation_ratio'] = 0.2
imp_hparams['test_ratio'] = 0.2
imp_hparams['val_check'] = 1
imp_hparams['project'] = 'practice'  # evaluation, Moco_training
imp_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
imp_hparams['quick_callback'] = True #True
training(imp_hparams)

print('SECOND RUN')
imp_hparams['bsz'] = 64
training(imp_hparams)

